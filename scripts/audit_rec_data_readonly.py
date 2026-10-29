#!/usr/bin/env python3
"""Standalone read-only recommendation data audit (no DPPBench imports).

Dependencies: Python >=3.9, pandas, pyarrow. Sources are opened for reading only.
Only reports and a temporary salted-hash SQLite index are written to output-dir.
No downloads, extraction, split construction, feature generation or training.
"""
import argparse
import collections
import datetime as dt
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import tempfile
import time

import pandas as pd
import pyarrow.parquet as pq


TASKS = ('amazon_beauty', 'kuairec', 'movielens', 'yelp')
ALIASES = {'amazonbeauty': 'amazon_beauty', 'allbeauty': 'amazon_beauty',
           'kuairec': 'kuairec', 'movielens': 'movielens', 'ml1m': 'movielens', 'yelp': 'yelp'}
PART_FILES = {'interaction_frozen.parquet': 'train_pool', 'train_frozen.parquet': 'train_pool',
              'train.parquet': 'train', 'val.parquet': 'val', 'validation.parquet': 'val',
              'val_frozen.parquet': 'val', 'validation_frozen.parquet': 'val',
              'std_test.parquet': 'std_test', 'test.parquet': 'test'}
LABELS = {'amazon_beauty': 'rating', 'movielens': 'rating', 'kuairec': 'watch_ratio', 'yelp': 'stars'}
USERS = ('user_id', 'reviewerID', 'userId', 'userid')
ITEMS = ('item_id', 'video_id', 'business_id', 'asin', 'movie_id', 'movieId')
TIMES = ('timestamp', 'unixReviewTime', 'date')
EVENTS = ('review_id', 'event_id', 'interaction_id')
TIME_METADATA = ('available_at', 'valid_from', 'snapshot_time', 'snapshot_date', 'updated_at', 'date')
USER_FIELDS = ('user_id', 'reviewerID', 'userId', 'user_gender', 'user_age', 'user_occupation',
               'user_zip_code', 'name', 'review_count', 'yelping_since', 'useful', 'funny',
               'cool', 'elite', 'friends', 'fans', 'average_stars', 'friend_list',
               'user_average_stars', 'user_review_count', 'user_fans', 'user_friends',
               'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
               'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
               'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range')
ITEM_FIELDS = ('item_id', 'video_id', 'business_id', 'asin', 'movie_id', 'movieId', 'brand',
               'item_brand', 'name', 'title', 'item_title', 'genres', 'item_genres', 'feat',
               'categories', 'item_categories', 'postal_code', 'stars', 'item_stars', 'is_open',
               'author_id', 'video_type', 'upload_type', 'visible_status', 'video_duration',
               'video_width', 'video_height', 'music_id', 'video_tag_id')
YELP_COUNTERS = ('review_count', 'user_review_count', 'average_stars', 'user_average_stars',
                 'stars', 'item_stars', 'useful', 'funny', 'cool', 'fans', 'user_fans')


def normalized(text):
    return ''.join(c for c in text.lower() if c.isalnum())


def missing(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


def canonical(value):
    if missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    return str(value)


def select(names, aliases):
    return next((c for c in aliases if c in names), None)


def time_value(value, calendar_date=False):
    """Return epoch seconds and an EXPLICIT inference tag; not available_at proof."""
    if missing(value):
        return None, 'missing'
    text = canonical(value)
    if calendar_date and len(text) == 8 and text.isdigit():
        try:
            stamp = dt.datetime.strptime(text, '%Y%m%d').replace(tzinfo=dt.timezone.utc)
            return stamp.timestamp(), 'calendar_YYYYMMDD_assumed_UTC'
        except ValueError:
            return None, 'invalid_calendar_date'
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        number = None
    if number is not None:
        if not math.isfinite(number):
            return None, 'invalid_nonfinite'
        size = abs(number)
        scale, unit = ((1e9, 'numeric_ns_inferred') if size >= 1e17 else
                       (1e6, 'numeric_us_inferred') if size >= 1e14 else
                       (1e3, 'numeric_ms_inferred') if size >= 1e11 else
                       (1., 'numeric_seconds_inferred' if size >= 1e8 else 'numeric_small_unit_ambiguous'))
        return number/scale, unit
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None, 'invalid_date'
        naive = stamp.tzinfo is None
        stamp = stamp.tz_localize('UTC') if naive else stamp.tz_convert('UTC')
        return stamp.timestamp(), 'date_naive_assumed_UTC' if naive else 'date_with_timezone'
    except (ValueError, TypeError, OverflowError):
        return None, 'invalid_date'


def identity_time(value, calendar_date=False):
    """Preserve integer nanoseconds in keys; float seconds are only for ranges."""
    if missing(value):
        return None
    text = canonical(value)
    if not (calendar_date and len(text) == 8 and text.isdigit()):
        try:
            number = Decimal(text)
            if not number.is_finite():
                return None
            size = abs(number)
            scale = (Decimal(10)**9 if size >= Decimal('1e17') else Decimal(10)**6
                     if size >= Decimal('1e14') else Decimal(10)**3 if size >= Decimal('1e11') else Decimal(1))
            return format((number/scale).normalize(), 'f')
        except InvalidOperation:
            pass
    try:
        if calendar_date and len(text) == 8 and text.isdigit():
            text = dt.datetime.strptime(text, '%Y%m%d').isoformat()
        stamp = pd.Timestamp(text)
        stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
        return format((Decimal(stamp.value)/Decimal(10)**9).normalize(), 'f')
    except (ValueError, TypeError, OverflowError):
        return None


def dat_columns(name):
    return (['user_id', 'user_gender', 'user_age', 'user_occupation', 'user_zip_code']
            if name == 'users.dat' else ['user_id', 'item_id', 'rating', 'timestamp']
            if name == 'ratings.dat' else ['item_id', 'item_title', 'item_genres'])


def observed_dtypes(info, frame, columns):
    observed = info.setdefault('observed_projected_dtypes', {})
    for c in columns:
        observed[c] = sorted(set(observed.get(c, [])) | {str(frame[c].dtype)})


def iso(value):
    try:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat() if value is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def summary(values):
    values = pd.Series(list(values), dtype='float64').dropna()
    if values.empty:
        return {'count': 0}
    return {'count': len(values), 'min': float(values.min()), 'median': float(values.median()),
            'p90': float(values.quantile(.9)), 'max': float(values.max()), 'mean': float(values.mean())}


def schema(path):
    lower = path.name.lower()
    if lower.endswith('.parquet'):
        p = pq.ParquetFile(path)
        return {f.name: str(f.type) for f in p.schema_arrow}, p.metadata.num_rows
    if lower.endswith('.csv'):
        return {c: 'not_scanned' for c in pd.read_csv(path, nrows=0).columns}, None
    if lower.endswith('.dat'):
        cols = dat_columns(lower)
        return {c: 'not_scanned' for c in cols}, None
    opener = gzip.open if lower.endswith('.gz') else open
    names = {}
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in itertools.islice(stream, 128):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('Expected JSON-lines object')
                for k, v in obj.items():
                    names[k] = type(v).__name__
    return names, None


def batches(path, columns, size):
    lower = path.name.lower()
    if lower.endswith('.parquet'):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns):
            yield batch.to_pandas()
    elif lower.endswith('.csv'):
        # Preserve large/zero-padded keys rather than inferring lossy floats.
        key_dtypes = {c: 'string' for c in columns if c in USERS or c in ITEMS}
        yield from pd.read_csv(path, usecols=columns, chunksize=size, low_memory=False, dtype=key_dtypes)
    else:
        opener = gzip.open if lower.endswith('.gz') else open
        encoding = 'latin-1' if lower.endswith('.dat') else 'utf-8'
        rows = []
        with opener(path, 'rt', encoding=encoding) as stream:
            for line in stream:
                if not line.strip():
                    continue
                if lower.endswith('.dat'):
                    keys = dat_columns(lower)
                    fields = line.rstrip('\r\n').split('::', len(keys)-1)
                    if len(fields) != len(keys):
                        raise ValueError('Malformed DAT row')
                    obj = dict(zip(keys, fields))
                else:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError('Expected JSON-lines object')
                rows.append({c: obj.get(c) for c in columns})
                if len(rows) >= size:
                    yield pd.DataFrame(rows, columns=columns)
                    rows = []
        if rows:
            yield pd.DataFrame(rows, columns=columns)


def discover(root):
    found, unassigned, symlinks = {t: [] for t in TASKS}, 0, 0
    generic_items = {'items.parquet', 'item.parquet', 'item_df.parquet', 'items.csv', 'item.csv'}
    generic_users = {'users.parquet', 'user.parquet', 'user_df.parquet', 'users.csv', 'user.csv'}
    sides = {'meta_all_beauty.json.gz': 'item_side', 'meta_all_beauty.json': 'item_side',
             'meta_all_beauty.jsonl': 'item_side', 'movies.dat': 'item_side', 'users.dat': 'user_side',
             'item_categories.csv': 'item_side', 'item_daily_features.csv': 'daily_side',
             'user_features.csv': 'user_side', 'social_network.csv': 'user_side',
             'yelp_academic_dataset_user.json': 'user_side',
             'yelp_academic_dataset_business.json': 'item_side'}
    raw = {'all_beauty.json.gz', 'ratings.dat', 'small_matrix.csv', 'big_matrix.csv',
           'yelp_academic_dataset_review.json', 'ml-1m.zip', 'kuairec.zip', 'yelp-json.zip'}
    for parent, dirs, names in os.walk(root, followlinks=False):
        symlinks += sum((Path(parent)/d).is_symlink() for d in dirs)
        dirs[:] = sorted(d for d in dirs if not (Path(parent)/d).is_symlink())
        for name in sorted(names):
            path = Path(parent)/name
            if path.is_symlink():
                symlinks += 1
                continue
            lower = name.lower()
            role = ('partition:'+PART_FILES[lower] if lower in PART_FILES else
                    'meta' if lower == 'meta.json' else 'item_side' if lower in generic_items else
                    'user_side' if lower in generic_users else sides.get(lower) or ('raw_inventory' if lower in raw else None))
            if role is None:
                continue
            parts = path.relative_to(root).parts[:-1]
            tasks = {ALIASES[normalized(p)] for p in parts if normalized(p) in ALIASES}
            # Also support a root explicitly set to a named dataset directory.
            if normalized(root.name) in ALIASES:
                tasks.add(ALIASES[normalized(root.name)])
            if len(tasks) != 1:
                unassigned += 1
                continue
            found[tasks.pop()].append((path, role))
    return found, {'recognized_files_without_unambiguous_task': unassigned,
                   'symlinks_not_followed': symlinks,
                   'policy': 'Enumerated existing files only; exact filename catalog and dataset directory aliases; no archive extraction or guessed fallback paths.'}


class Audit:
    def __init__(self, db, batch_size):
        self.db, self.batch_size, self.salt = db, batch_size, secrets.token_bytes(32)
        self.db.executescript('''
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-32768;
        CREATE TABLE events(src INTEGER, part TEXT, k BLOB, e BLOB, u BLOB, i BLOB, t REAL, label REAL);
        CREATE TABLE side(src INTEGER, k BLOB, first BLOB, n INTEGER, conflict INTEGER, PRIMARY KEY(src,k));
        CREATE TABLE daily(src INTEGER, k BLOB, day REAL, n INTEGER, first BLOB, conflict INTEGER, PRIMARY KEY(src,k,day));
        CREATE TABLE changes(src INTEGER, k BLOB, field TEXT, first BLOB, changed INTEGER, PRIMARY KEY(src,k,field));
        ''')

    def digest(self, *parts):
        if any(p is None for p in parts):
            return None
        return hashlib.sha256(self.salt+json.dumps(parts, sort_keys=True, default=str,
            ensure_ascii=False, separators=(',', ':')).encode('utf-8')).digest()

    def fingerprint(self, path):
        stat = path.stat()
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(chunk)
        return {'path': str(path), 'bytes': stat.st_size, 'sha256': h.hexdigest(),
                'mtime_ns_before': stat.st_mtime_ns}

    def scan(self, src, task, path, role):
        info = self.fingerprint(path)
        info.update(source_id=src, role=role, status='ok')
        self.current_info = info
        if role == 'meta':
            # Never echo arbitrary metadata values/ID lists.
            obj = json.loads(path.read_text(encoding='utf-8'))
            info['field_names'] = list(obj) if isinstance(obj, dict) else []
            info['schema'] = {k: type(v).__name__ for k, v in obj.items()} if isinstance(obj, dict) else {}
            info['known_protocol_scalars'] = {k: obj[k] for k in
                ('kind', 'seed', 'test_size', 'split_method', 'label_source', 'std_test_positives', 'std_test_negatives')
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], (str, int, float, bool))}
        elif role == 'raw_inventory':
            if not path.name.lower().endswith('.zip'):
                cols, _ = schema(path)
                info['schema'] = cols
            info['status'] = 'inventory_only_not_scanned_as_model_input'
            info['rows'] = None
        else:
            cols, rows = schema(path)
            info.update(schema=cols, parquet_rows=rows,
                        schema_scope='full Parquet/CSV header; JSON first 128 lines, projected fields scanned fully')
            if role.startswith('partition:'):
                self.interactions(src, task, path, role.split(':')[1], info)
            else:
                self.side(src, path, role, info)
        stat = path.stat()
        info['unchanged_during_read'] = stat.st_size == info['bytes'] and stat.st_mtime_ns == info['mtime_ns_before']
        if not info['unchanged_during_read']:
            info['status'] = 'source_changed_during_audit_invalid_snapshot'
        return info

    def interactions(self, src, task, path, partition, info):
        cols = info['schema']
        user, item, clock, event = select(cols, USERS), select(cols, ITEMS), select(cols, TIMES), select(cols, EVENTS)
        target = LABELS[task] if LABELS[task] in cols else None
        alternatives = [c for c in ('label', 'target', 'rating', 'stars', 'watch_ratio') if c in cols]
        if target is None and len(alternatives) == 1:
            target = alternatives[0]
        chosen = list(dict.fromkeys(c for c in (user, item, clock, event, target, '__split__') if c and c in cols))
        info['projection'] = chosen
        info['file_declared_partition'] = partition
        info['marker_vs_filename_conflict_rows'] = 0
        info['identity_definition'] = {'candidate_key': [user, item, clock], 'native_event_column': event,
            'label_column': target, 'label_candidates': alternatives,
            'warning': 'user+item+normalized time is a candidate identity, not proof of same event; coarse timestamps/multiple events can collide. Native event IDs require a documented globally unique namespace; same column name alone cannot certify that. IDs are never emitted.'}
        if not chosen:
            info.update(rows=info.get('parquet_rows'), status='unverified_no_recognized_audit_columns')
            return
        stats = {}
        for frame in batches(path, chosen, self.batch_size):
            observed_dtypes(info, frame, chosen)
            records = []
            for row in frame.to_dict('records'):
                marker = row.get('__split__', partition)
                part = {'train': 'train', 'val': 'val', 'valid': 'val', 'validation': 'val',
                        'test': 'test', 'std_test': 'std_test', 'train_pool': 'train_pool'}.get(str(marker), 'unknown_partition')
                allowed = {'train_pool', 'train', 'val'} if partition == 'train_pool' else {partition}
                info['marker_vs_filename_conflict_rows'] += int(part not in allowed)
                s = stats.setdefault(part, {'rows': 0, 'nulls': collections.Counter(), 'label_counts': collections.Counter(),
                    'label_null': 0, 'label_non_numeric': 0, 'time_tags': collections.Counter(),
                    'time_min': None, 'time_max': None, 'incomplete_candidate_identity': 0})
                s['rows'] += 1
                for c in chosen:
                    s['nulls'][c] += int(missing(row.get(c)))
                u, i = canonical(row.get(user)), canonical(row.get(item))
                t, tag = time_value(row.get(clock), calendar_date=clock == 'date')
                s['time_tags'][tag] += 1
                if t is not None:
                    s['time_min'] = t if s['time_min'] is None else min(t, s['time_min'])
                    s['time_max'] = t if s['time_max'] is None else max(t, s['time_max'])
                k = self.digest('interaction', u, i, identity_time(row.get(clock), calendar_date=clock == 'date'))
                s['incomplete_candidate_identity'] += int(k is None)
                e = self.digest('event', event, canonical(row.get(event))) if event else None
                label = None
                if target is not None:
                    if missing(row.get(target)):
                        s['label_null'] += 1
                    else:
                        try:
                            label = float(row[target])
                            if not math.isfinite(label):
                                raise ValueError('nonfinite')
                            # Rating/target values only; never echo unexpected string labels.
                            key = str(label) if len(s['label_counts']) < 32 or str(label) in s['label_counts'] else 'OTHER_NUMERIC'
                            s['label_counts'][key] += 1
                        except (ValueError, TypeError, OverflowError):
                            s['label_non_numeric'] += 1
                records.append((src, part, k, e, self.digest('user', u), self.digest('item', i), t, label))
            self.db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?,?)', records)
            self.db.commit()
        for s in stats.values():
            s['time_min_utc_assuming_inferred_units'] = iso(s['time_min'])
            s['time_max_utc_assuming_inferred_units'] = iso(s['time_max'])
        info['partitions'] = stats
        info['rows'] = sum(s['rows'] for s in stats.values())
        info['validation_note'] = 'Only materialized file/markers audited. No seed-specific runtime validation split was reconstructed; train_pool may later be divided into train/val.'

    def side(self, src, path, role, info):
        cols = info['schema']
        key = select(cols, USERS if role == 'user_side' else ITEMS)
        wanted = USER_FIELDS if role == 'user_side' else ITEM_FIELDS
        chosen = [c for c in cols if c == key or c in wanted or c in TIME_METADATA or c.startswith('compliment_')]
        info.update(key_column=key, projection=chosen, rows=0, null_key_rows=0,
                    nulls=collections.Counter(), numeric_summaries={}, date_parse_tags=collections.Counter(),
                    conflict_scope='Projected loader-relevant attributes only; unprojected attributes NOT certified')
        if not key:
            info['status'] = 'unverified_missing_recognized_key'
            return
        numeric = {}
        for frame in batches(path, chosen, self.batch_size):
            observed_dtypes(info, frame, chosen)
            for c in chosen:
                info['nulls'][c] += int(frame[c].map(missing).sum())
                if c in YELP_COUNTERS:
                    vals = pd.to_numeric(frame[c], errors='coerce').dropna()
                    vals = vals[vals.map(math.isfinite)]
                    if len(vals):
                        s = numeric.setdefault(c, {'count': 0, 'min': math.inf, 'max': -math.inf, 'sum': 0.})
                        s.update(count=s['count']+len(vals), min=min(s['min'], float(vals.min())),
                                 max=max(s['max'], float(vals.max())), sum=s['sum']+float(vals.sum()))
            sides, dates, changes = [], [], []
            for row in frame.to_dict('records'):
                info['rows'] += 1
                token = canonical(row.get(key))
                if token is None:
                    info['null_key_rows'] += 1
                    continue
                kh = self.digest('user' if role == 'user_side' else 'item', token)
                fingerprint = self.digest('attributes', {c: canonical(row[c]) for c in chosen if c != key})
                sides.append((src, kh, fingerprint))
                if role == 'daily_side':
                    day, tag = time_value(row.get('date'), calendar_date=True)
                    info['date_parse_tags'][tag] += 1
                    if day is not None:
                        dates.append((src, kh, day, fingerprint))
                    for c in chosen:
                        if c != key and c not in TIME_METADATA:
                            changes.append((src, kh, c, self.digest('field', canonical(row[c]))))
            self.db.executemany('''INSERT INTO side VALUES (?,?,?,1,0) ON CONFLICT(src,k)
                DO UPDATE SET n=n+1, conflict=MAX(conflict,side.first!=excluded.first)''', sides)
            self.db.executemany('''INSERT INTO daily VALUES (?,?,?,1,?,0) ON CONFLICT(src,k,day)
                DO UPDATE SET n=n+1, conflict=MAX(conflict,daily.first!=excluded.first)''', dates)
            self.db.executemany('''INSERT INTO changes VALUES (?,?,?,?,0) ON CONFLICT(src,k,field)
                DO UPDATE SET changed=MAX(changed,changes.first IS NOT excluded.first)''', changes)
            self.db.commit()
        counts = self.db.execute('SELECT COUNT(*),SUM(n>1),SUM(conflict),SUM(n-1) FROM side WHERE src=?', (src,)).fetchone()
        info['key_audit'] = dict(zip(('unique_keys', 'duplicate_keys', 'conflicting_keys', 'extra_rows_same_key'), (v or 0 for v in counts)))
        info['numeric_summaries'] = {c: {'count': s['count'], 'min': s['min'], 'max': s['max'],
                                          'mean': s['sum']/s['count']} for c, s in numeric.items()}
        info['availability_field_names_present'] = [c for c in TIME_METADATA if c in cols]
        info['availability_verdict'] = 'UNVERIFIED: values and field names cannot prove snapshot generation time or that val/std_test reviews were excluded.'
        if role == 'daily_side':
            coverage = self.db.execute('SELECT COUNT(*),MIN(day),MAX(day) FROM daily WHERE src=? GROUP BY k', (src,)).fetchall()
            info['daily_coverage'] = {'items_with_parseable_dates': len(coverage),
                'distinct_dates_per_item': summary(r[0] for r in coverage),
                'coverage_span_days': summary((r[2]-r[1])/86400 for r in coverage),
                'global_min_date': iso(min((r[1] for r in coverage), default=None)),
                'global_max_date': iso(max((r[2] for r in coverage), default=None)),
                'items_with_changed_field': dict(self.db.execute('SELECT field,SUM(changed) FROM changes WHERE src=? GROUP BY field', (src,))),
                'interpretation': 'Observed snapshot date coverage and value changes only. Date midnight (UTC assumption) is NOT the actual available_at. Same-day snapshots may contain later observations.'}
            counts = self.db.execute('SELECT COUNT(*),SUM(n>1),SUM(conflict) FROM daily WHERE src=?', (src,)).fetchone()
            info['daily_date_key_audit'] = dict(zip(('unique_item_date_keys', 'repeated_item_date_keys',
                'conflicting_item_date_keys'), (n or 0 for n in counts)))
            info['key_uniqueness_note'] = 'Repeated item IDs across dates are expected time-series structure, not an invalid side key by themselves. Inspect item+date conflicts separately. Latest-snapshot collapsing is a modeling choice, not performed here.'

    def comparisons(self, sources):
        ids = [s['source_id'] for s in sources if s['status'] == 'ok' and s['role'].startswith('partition:') and 'partitions' in s]
        self.db.executescript('CREATE INDEX IF NOT EXISTS e_key ON events(src,part,k); CREATE INDEX IF NOT EXISTS e_native ON events(src,part,e); CREATE INDEX IF NOT EXISTS e_item ON events(src,i); CREATE INDEX IF NOT EXISTS d_item ON daily(src,k);')
        result = {'within_partitions': [], 'between_partitions': [], 'join_coverage': [], 'daily_vs_queries': [],
                  'excluded_incomplete_or_changed_sources': [s['source_id'] for s in sources if s['status'] in
                      ('failed_or_partial', 'source_changed_during_audit_invalid_snapshot')]}
        partitions = [(src, part) for src in ids for part, in self.db.execute('SELECT DISTINCT part FROM events WHERE src=?', (src,))]
        for src, part in partitions:
            q = '''SELECT COUNT(*),SUM(n>1),SUM(labels>1),SUM(native>1) FROM
                (SELECT k,COUNT(*) n,COUNT(DISTINCT label) labels,COUNT(DISTINCT e) native
                 FROM events WHERE src=? AND part=? AND k IS NOT NULL GROUP BY k)'''
            row = self.db.execute(q, (src, part)).fetchone()
            result['within_partitions'].append({'source_id': src, 'partition': part,
                **dict(zip(('unique_complete_candidate_identities', 'repeated_candidate_identities',
                            'identities_with_multiple_labels', 'identities_with_multiple_native_event_ids'), (x or 0 for x in row))),
                'verdict': 'Repeated composite keys are ambiguous; multiple native event IDs demonstrate possible distinct events at same user/item/time.'})
        for (a, pa), (b, pb) in itertools.combinations(partitions, 2):
            row = self.db.execute('''WITH x AS (SELECT k,MIN(label) lo,MAX(label) hi FROM events
                WHERE src=? AND part=? AND k IS NOT NULL GROUP BY k),
                y AS (SELECT k,MIN(label) lo,MAX(label) hi FROM events
                WHERE src=? AND part=? AND k IS NOT NULL GROUP BY k)
                SELECT COUNT(*),SUM(x.lo!=y.hi OR x.hi!=y.lo) FROM x JOIN y ON x.k=y.k''', (a, pa, b, pb)).fetchone()
            native = self.db.execute('''SELECT COUNT(*) FROM
                (SELECT DISTINCT e FROM events WHERE src=? AND part=? AND e IS NOT NULL) x JOIN
                (SELECT DISTINCT e FROM events WHERE src=? AND part=? AND e IS NOT NULL) y ON x.e=y.e''', (a, pa, b, pb)).fetchone()[0]
            result['between_partitions'].append({'left': [a, pa], 'right': [b, pb],
                'shared_candidate_identities': row[0], 'candidate_identities_with_label_disagreement': row[1] or 0,
                'shared_native_event_ids_same_named_namespace': native,
                'interpretation': 'A candidate-key intersection is an audit flag, not automatic proof of duplicate interactions or label leakage. Missing identities are excluded, not declared disjoint.'})
        for side in sources:
            if side['status'] != 'ok' or 'key_audit' not in side:
                continue
            sid = side['source_id']
            col = 'u' if side['role'] == 'user_side' else 'i'
            for src, part in partitions:
                row = self.db.execute(f'''SELECT COUNT(*),SUM(e.{col} IS NULL),SUM(e.{col} IS NOT NULL AND s.k IS NULL)
                    FROM events e LEFT JOIN side s ON s.src=? AND s.k=e.{col} WHERE e.src=? AND e.part=?''', (sid, src, part)).fetchone()
                total, nulls, absent = (v or 0 for v in row)
                result['join_coverage'].append({'side_source_id': sid, 'interaction_source_id': src, 'partition': part,
                    'rows': total, 'null_interaction_key_rows': nulls, 'unmatched_nonnull_key_rows': absent,
                    'missing_association_fraction': (nulls+absent)/total if total else None,
                    'note': 'Coverage of this individual side table, not a guessed merged loader table.'})
                if side['role'] == 'daily_side':
                    row = self.db.execute('''SELECT COUNT(*),SUM(e.t IS NULL),SUM(e.t IS NOT NULL AND d.last IS NULL),
                        SUM(d.last>e.t),SUM(CAST(d.last/86400 AS INT)=CAST(e.t/86400 AS INT))
                        FROM events e LEFT JOIN (SELECT k,MAX(day) last FROM daily WHERE src=? GROUP BY k) d
                        ON d.k=e.i WHERE e.src=? AND e.part=?''', (sid, src, part)).fetchone()
                    result['daily_vs_queries'].append({'daily_source_id': sid, 'interaction_source_id': src, 'partition': part,
                        **dict(zip(('query_rows', 'missing_query_time_rows', 'missing_snapshot_rows',
                                    'last_snapshot_date_midnight_after_query_rows', 'same_calendar_day_rows'), (x or 0 for x in row))),
                        'verdict': 'Date comparison only, using reported time-unit/UTC assumptions. Earlier date does not establish real availability; same-day needs actual release time.'})
        return result


def markdown(report):
    lines = ['# Recommendation data read-only audit', '',
        'This is a data inventory/audit, NOT experiment clearance. No downloads, splits, data edits or model training.', '',
        'Privacy: aggregate statistics only. Raw IDs, comment contents, row-level records and identity hashes are not emitted.', '',
        'Identity: user + item + normalized timestamp is only a candidate event key. Native event IDs are checked separately when available. Shared user/item alone is NOT an interaction duplicate.', '',
        'Validation: only existing validation files/markers are audited; runtime validation is not recreated.', '',
        'Time: numeric units and naive UTC parsing are explicitly inferred. Snapshot date is NOT available_at.', '']
    for task, result in report['datasets'].items():
        lines += ['## '+task, '', '| Source | Role | Rows | Bytes | Status |', '|---|---|---:|---:|---|']
        for s in result['sources']:
            path = s['path'].replace('|', '\\|')
            lines.append(f"| {path} | {s['role']} | {s.get('rows', 'unknown')} | {s.get('bytes', 'unknown')} | {s['status']} |")
        lines += ['', 'Missing/unverified: '+', '.join(result['missing_or_unverified']), '',
                  '### Aggregated findings', '', '```json',
                  json.dumps(result['comparisons'], ensure_ascii=False, indent=2), '```', '',
                  '### Provenance and schema evidence', '',
                  'SHA256, full reported schema, projected-field dtypes/nulls, partition label/time distributions and side-key conflicts are in the companion JSON.', '']
        for s in result['sources']:
            lines += ['- source '+str(s['source_id'])+': SHA256 `'+s.get('sha256', 'unavailable')+'`']
            if s.get('schema'):
                lines += ['', '```json', json.dumps({'schema': s['schema'],
                    'observed_projected_dtypes': s.get('observed_projected_dtypes', {}),
                    'projection': s.get('projection'), 'nulls_projected_fields': s.get('nulls')},
                    ensure_ascii=False, indent=2), '```']
            for field in ('partitions', 'key_audit', 'daily_coverage', 'numeric_summaries'):
                if s.get(field):
                    lines += ['', '```json', json.dumps({field: s[field]}, ensure_ascii=False, indent=2), '```']
        lines += ['', 'Availability verdict: '+result['availability_verdict'], '']
    lines += ['## Limitations', '',
        '- Multiple matching files are all inventoried, never silently chosen or concatenated as a certified dataset. Source IDs identify pairwise comparisons.',
        '- Duplicate conflicts cover the explicitly projected attributes. Unknown/unprojected fields and unsupported filenames are not certified.',
        '- Yelp snapshot averages/counts do not encode their generation time or constituent review identities. Matching ranges/values cannot prove absence of held-out label contribution.',
        '- KuaiRec daily coverage/changes and later-date counts are evidence of snapshot selection risk, not proof that every field is mutable.',
        '- No per-user IDs or per-item histories are printed; per-item coverage is summarized across items.',
        '- The temporary SQLite index stores only salted identity/attribute hashes plus audit time/label scalars, is deleted on exit, and is not a report artifact.',
        '- Large audits need free disk space in output-dir for temporary indexes; RAM is bounded by batches and a 32 MiB SQLite cache, plus per-item KuaiRec coverage summaries. SHA256 streams the entire source bytes, including large raw files.',
        '- Sources are not locked. Files changed during reading invalidate their snapshot audit; review unchanged_during_read in JSON.', '']
    return '\n'.join(lines)


def write_report(path, text):
    """Atomic replacement also avoids following an existing report-file symlink."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                prefix='.rec_report_', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/sdb/dengjiale/dppbench/data'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    root, out = args.data_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error('--data-root must be an existing directory')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if out == root or root in out.parents:
        parser.error('--output-dir must be outside --data-root (no writes into the input tree)')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    discovered, discovery = discover(root)
    report = {'audit_version': 1, 'data_root': str(root), 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'read_only_sources': True, 'full_search_or_hpo_run': False, 'formal_experiments_cleared': False,
              'dependencies': {'python': sys.version.split()[0], 'pandas': pd.__version__,
                               'pyarrow': sys.modules['pyarrow'].__version__},
              'discovery': discovery, 'datasets': {}}
    with tempfile.TemporaryDirectory(prefix='.rec_audit_scratch_', dir=out) as scratch:
        db = sqlite3.connect(str(Path(scratch)/'index.sqlite'))
        try:
            audit, source_id = Audit(db, args.batch_size), 0
            for task in TASKS:
                sources = []
                for path, role in discovered[task]:
                    source_id += 1
                    print(f'[{task}] reading source {source_id}: {path.name}', flush=True)
                    try:
                        sources.append(audit.scan(source_id, task, path, role))
                    except Exception as error:
                        # No error message/traceback that might contain source row contents.
                        partial = dict(audit.current_info) if getattr(audit, 'current_info', {}).get('source_id') == source_id else {}
                        partial.update(source_id=source_id, path=str(path), role=role,
                                       status='failed_or_partial', error_type=type(error).__name__)
                        sources.append(partial)
                roles = collections.Counter(s['role'] for s in sources)
                missing_items = [r for r in ('partition:train_pool', 'partition:std_test', 'meta', 'item_side', 'user_side') if not roles[r]]
                if not any(s.get('partitions', {}).get('val', {}).get('rows', 0) > 0 for s in sources):
                    missing_items.append('materialized validation: absent; runtime validation membership UNVERIFIED')
                if task == 'kuairec' and not roles['daily_side']:
                    missing_items.append('item_daily_features')
                availability = ('BLOCKED pending available_at/point-in-time provenance. Latest daily snapshots and user/social snapshots require temporal evidence.'
                    if task == 'kuairec' else 'BLOCKED pending snapshot generation dates and review membership. Averages/counts alone cannot establish train-only provenance.'
                    if task == 'yelp' else 'UNVERIFIED: static metadata/key availability and dataset identity/time checks require review; no automatic clearance.')
                report['datasets'][task] = {'sources': sources, 'missing_or_unverified': missing_items,
                    'multiple_candidates_by_role': {r: n for r, n in roles.items() if n > 1},
                    'comparisons': audit.comparisons(sources), 'availability_verdict': availability}
        finally:
            db.close()
    report['elapsed_seconds'] = round(time.monotonic()-started, 3)
    json_path, md_path = out/'rec_data_audit.json', out/'rec_data_audit.md'
    write_report(json_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write_report(md_path, markdown(report))
    print(f'Reports: {json_path}\n         {md_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

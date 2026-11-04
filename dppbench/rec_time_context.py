"""Opaque row-aligned event context; never a predictor or fitted statistic."""
from dataclasses import dataclass


EVENT_CONTEXT = '__dppbench_rec_event_context__'


def time_rule(data, column):
    feature = (data.model_cfg or {}).get('feature', {})
    return source_time_rule(data.name, column, feature.get('time_rules', {}))


def source_time_rule(name, column, rules=None):
    """Explicit source contracts, never units inferred from value magnitudes.

    Amazon unixReviewTime, MovieLens Unix timestamps, and Yelp's loader output
    (datetime int64 // 1e9) are seconds. KuaiRec's official field contract
    declares Unix timestamps with fractional seconds (e.g. 1593878903.438):
    https://github.com/chongminggao/KuaiRec#data-descriptions
    """
    explicit = (rules or {}).get(column)
    if explicit:
        return explicit
    names = {'AmazonBeauty', 'amazon_beauty', 'Movielens', 'MovieLens', 'movielens',
             'Yelp', 'yelp', 'KuaiRec', 'Kuairec', 'kuairec'}
    if column == 'timestamp' and name in names:
        return {'unit': 's', 'origin': 'unix'}
    return None


@dataclass(frozen=True)
class EventContext:
    user: object
    time: object


def preserve_event_context(frame, user_col, time_col):
    """Capture once, before a candidate changes the event clock.

    Nonnumeric scalars travel with rows through sorting/filtering/replication.
    The candidate's visible timestamp remains free to change representation.
    """
    if EVENT_CONTEXT in frame:
        event_context(frame)
        return frame
    out = frame.copy(deep=False)
    out[EVENT_CONTEXT] = [EventContext(user, time) for user, time in
                          frame[[user_col, time_col]].itertuples(index=False, name=None)]
    return out


def event_context(frame):
    values = frame[EVENT_CONTEXT].tolist()
    if any(not isinstance(value, EventContext) for value in values):
        raise ValueError('Recommendation event context was lost or modified; '
                         'cannot assign an observed history clock to synthetic rows')
    return values


def original_queries(frame, user_col, time_col):
    """Resolve positional query keys without altering candidate features."""
    if EVENT_CONTEXT not in frame:
        return frame[user_col].tolist(), frame[time_col].tolist()
    values = event_context(frame)
    return [v.user for v in values], [v.time for v in values]

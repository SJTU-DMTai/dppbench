"""Logical-pipeline search via genetic algorithm.

This is SAGA's "logical" level: we search over the *structure* of a
preprocessing pipeline (which operators are present and in what order). The
hyperparameters of each operator are kept at their default values during this
phase; they are tuned later by ``physical_search.PhysicalSearch``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from baselines.common.evaluator import EvaluationResult, PipelineEvaluator
from .operator_catalog import CATALOG, operators_for_task
from baselines.common.pipeline import (
    DataContext,
    Pipeline,
    PipelineStep,
    assign_dag_structure,
    make_step,
    random_pipeline,
)
from baselines.common.pipeline_constraints import is_legal, repair


@dataclass
class GAIndividual:
    pipeline: Pipeline
    fitness: float = -math.inf
    metrics: dict = field(default_factory=dict)


class LogicalSearch:
    def __init__(
        self,
        evaluator: PipelineEvaluator,
        ctx: DataContext,
        population_size: int = 12,
        n_generations: int = 5,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.6,
        tournament_size: int = 3,
        elitism_size: int = 2,
        early_stop_patience: int = 3,
        random_pipeline_p_optional: float = 0.5,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.evaluator = evaluator
        self.ctx = ctx
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.early_stop_patience = early_stop_patience
        self.random_pipeline_p_optional = float(random_pipeline_p_optional)
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def run(self, top_k: int = 3) -> list[GAIndividual]:
        # 1. Initial population
        population: list[GAIndividual] = []
        attempts = 0
        while len(population) < self.population_size and attempts < self.population_size * 5:
            attempts += 1
            pipe = random_pipeline(
                self.ctx.task_type,
                self.ctx,
                self.rng,
                p_optional=self.random_pipeline_p_optional,
            )
            ind = self._evaluate(pipe)
            if ind.fitness > -math.inf:
                population.append(ind)
        if not population:
            raise RuntimeError(
                "Could not generate any successful pipeline during initialisation."
            )
        if self.verbose:
            print(
                f"[GA] init  population={len(population)}  "
                f"best={max(p.fitness for p in population):.4f}"
            )
        self._record_generation(0, population)

        best_fitness_so_far = max(ind.fitness for ind in population)
        no_improve = 0

        # 2. Generations
        for gen in range(1, self.n_generations + 1):
            population = self._next_generation(population)
            cur_best = max(ind.fitness for ind in population)
            improved = cur_best > best_fitness_so_far + 1e-6
            if improved:
                best_fitness_so_far = cur_best
                no_improve = 0
            else:
                no_improve += 1
            if self.verbose:
                print(
                    f"[GA] gen={gen}  best={cur_best:.4f}  "
                    f"avg={sum(p.fitness for p in population)/len(population):.4f}  "
                    f"no_improve={no_improve}"
                )
            self._record_generation(gen, population)
            if no_improve >= self.early_stop_patience:
                if self.verbose:
                    print(f"[GA] early stop after {gen} generations")
                break

        # 3. Top-K
        population.sort(key=lambda i: i.fitness, reverse=True)
        # de-duplicate by pipeline hash
        seen, dedup = set(), []
        for ind in population:
            h = ind.pipeline.hash()
            if h in seen:
                continue
            seen.add(h)
            dedup.append(ind)
            if len(dedup) >= top_k:
                break
        return dedup

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def _evaluate(self, pipeline: Pipeline) -> GAIndividual:
        ev: EvaluationResult = self.evaluator.evaluate(pipeline)
        return GAIndividual(pipeline=pipeline, fitness=ev.fitness, metrics=ev.metrics)

    def _record_generation(self, gen: int, population: list[GAIndividual]) -> None:
        fits = [p.fitness for p in population if p.fitness > -math.inf]
        self.history.append({
            "generation": gen,
            "size": len(population),
            "best": max(fits) if fits else -math.inf,
            "avg": (sum(fits) / len(fits)) if fits else -math.inf,
        })

    def _tournament_select(self, population: list[GAIndividual]) -> GAIndividual:
        contenders = self.rng.sample(
            population, k=min(self.tournament_size, len(population))
        )
        return max(contenders, key=lambda i: i.fitness)

    def _next_generation(self, population: list[GAIndividual]) -> list[GAIndividual]:
        # Elitism
        population.sort(key=lambda i: i.fitness, reverse=True)
        new_pop: list[GAIndividual] = list(population[: self.elitism_size])

        while len(new_pop) < self.population_size:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)

            child_pipe = parent_a.pipeline.copy()
            if self.rng.random() < self.crossover_rate:
                child_pipe = self._crossover(parent_a.pipeline, parent_b.pipeline)
            if self.rng.random() < self.mutation_rate:
                child_pipe = self._mutate(child_pipe)

            repair(child_pipe, self.ctx.task_type, self.ctx)
            assign_dag_structure(child_pipe, self.ctx, self.rng)
            ind = self._evaluate(child_pipe)
            if ind.fitness > -math.inf:
                new_pop.append(ind)
            else:
                # fallback: keep elite to avoid stalling
                new_pop.append(self.rng.choice(population[: max(1, self.elitism_size)]))

        return new_pop[: self.population_size]

    def _crossover(self, a: Pipeline, b: Pipeline) -> Pipeline:
        # one-point crossover by position
        if not a.steps or not b.steps:
            return a.copy()
        cut_a = self.rng.randint(0, len(a.steps))
        cut_b = self.rng.randint(0, len(b.steps))
        new_steps = [PipelineStep.from_dict(s.to_dict()) for s in a.steps[:cut_a]]
        new_steps += [PipelineStep.from_dict(s.to_dict()) for s in b.steps[cut_b:]]
        return Pipeline(steps=new_steps)

    def _mutate(self, p: Pipeline) -> Pipeline:
        ops = operators_for_task(self.ctx.task_type)
        # filter out mandatory & always-included ops to avoid dropping them
        choice = self.rng.choice(["add", "remove", "replace", "swap", "rewire"])
        if choice == "add" or not p.steps:
            op_name = self.rng.choice(ops)
            step = make_step(op_name, self.ctx, self.rng)
            if step is not None:
                p.steps.append(step)
        elif choice == "remove" and len(p.steps) > 1:
            idx = self.rng.randrange(len(p.steps))
            if not CATALOG[p.steps[idx].op].mandatory:
                p.steps.pop(idx)
        elif choice == "replace":
            idx = self.rng.randrange(len(p.steps))
            if not CATALOG[p.steps[idx].op].mandatory:
                op_name = self.rng.choice(ops)
                step = make_step(op_name, self.ctx, self.rng)
                if step is not None:
                    p.steps[idx] = step
        elif choice == "swap" and len(p.steps) >= 2:
            i, j = self.rng.sample(range(len(p.steps)), 2)
            p.steps[i], p.steps[j] = p.steps[j], p.steps[i]
        elif choice == "rewire":
            assign_dag_structure(p, self.ctx, self.rng)
        return p

//! Integration tests: each analytics pack as a Suggestor in the convergence loop.

use converge_kernel::{Budget, ContextFact, ContextKey, ContextState, ConvergeResult, Engine};
use converge_pack::{PackInputPayload, PackPlanPayload, PackSuggestor, ProposedFact};
use prism::packs::fuzzy::{SugenoInferencePack, TsukamotoInferencePack};
use prism::packs::{
    AnomalyDetectionPack, ClassificationPack, DescriptiveStatsPack, ForecastingPack,
    FuzzyInferencePack, NaiveBayesPack, RankingPack, RegressionPack, SegmentationPack,
    SimilarityPack, TrendDetectionPack,
};

fn budget() -> Budget {
    Budget {
        max_cycles: 5,
        max_facts: 100,
    }
}

async fn run_with_input<P: converge_pack::Pack + 'static>(
    pack: P,
    input: serde_json::Value,
) -> ConvergeResult {
    let pack_name = pack.name();
    let mut engine = Engine::with_budget(budget());
    engine.register_suggestor(PackSuggestor::new(
        pack,
        ContextKey::Seeds,
        ContextKey::Strategies,
    ));
    let mut ctx = ContextState::new();
    add_pack_seed(&mut ctx, "input-1", pack_name, input);
    engine.run(ctx).await.expect("should converge")
}

fn add_pack_seed(ctx: &mut ContextState, id: &str, pack: &str, input: serde_json::Value) {
    ctx.add_proposal(ProposedFact::new(
        ContextKey::Seeds,
        id,
        PackInputPayload::new(pack, input),
        "test",
    ))
    .expect("pack seed proposal should stage");
}

fn fact_content(fact: &ContextFact) -> String {
    fact.text()
        .map(str::to_string)
        .or_else(|| {
            fact.payload::<PackPlanPayload>()
                .map(|payload| payload.plan.to_string())
        })
        .or_else(|| {
            fact.to_wire()
                .ok()
                .map(|wire| wire.payload.payload.to_string())
        })
        .expect("fact payload should render for assertion")
}

#[tokio::test]
async fn anomaly_detection_finds_outliers() {
    let result = run_with_input(
        AnomalyDetectionPack,
        serde_json::json!({
            "values": [10.0, 11.0, 10.5, 9.8, 10.2, 50.0, 10.1, 9.9, 10.3, 100.0],
            "threshold": 2.0
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("anomalies"));
    assert!(content.contains("50.0") || content.contains("100.0"));
}

#[tokio::test]
async fn segmentation_clusters_distinct_groups() {
    let result = run_with_input(
        SegmentationPack,
        serde_json::json!({
            "records": [
                [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
                [10.0, 10.0], [10.1, 9.9], [9.9, 10.1]
            ],
            "k": 2
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("assignments"));
    assert!(content.contains("centroids"));
}

#[tokio::test]
async fn ranking_orders_by_composite_score() {
    let result = run_with_input(
        RankingPack,
        serde_json::json!({
            "items": [
                {"id": "worst", "scores": [0.1, 0.1]},
                {"id": "best", "scores": [0.9, 0.9]},
                {"id": "middle", "scores": [0.5, 0.5]}
            ],
            "weights": [0.5, 0.5],
            "higher_is_better": [true, true]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    // "best" should appear before "worst" in ranked output
    let best_pos = content.find("best").expect("best should be in output");
    let worst_pos = content.find("worst").expect("worst should be in output");
    assert!(best_pos < worst_pos, "best should rank before worst");
}

#[tokio::test]
async fn forecasting_produces_predictions() {
    let result = run_with_input(
        ForecastingPack,
        serde_json::json!({
            "values": [100.0, 110.0, 120.0, 130.0, 140.0],
            "horizon": 3,
            "alpha": 0.3
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("predictions"));
    assert!(content.contains("\"step\":1"));
    assert!(content.contains("\"step\":2"));
    assert!(content.contains("\"step\":3"));
}

#[tokio::test]
async fn fuzzy_inference_grades_expectation_signal() {
    let result = run_with_input(
        FuzzyInferencePack,
        serde_json::json!({
            "inputs": {
                "authenticity": 0.84,
                "novelty": 0.25,
                "wait_time": 0.40
            },
            "variables": [
                {
                    "name": "authenticity",
                    "sets": [
                        {"name": "low", "function": {"kind": "left_shoulder", "start": 0.2, "end": 0.5}},
                        {"name": "high", "function": {"kind": "right_shoulder", "start": 0.6, "end": 0.9}}
                    ]
                },
                {
                    "name": "novelty",
                    "sets": [
                        {"name": "low", "function": {"kind": "left_shoulder", "start": 0.2, "end": 0.6}},
                        {"name": "high", "function": {"kind": "right_shoulder", "start": 0.5, "end": 0.9}}
                    ]
                },
                {
                    "name": "wait_time",
                    "sets": [
                        {"name": "low", "function": {"kind": "left_shoulder", "start": 0.2, "end": 0.6}},
                        {"name": "high", "function": {"kind": "right_shoulder", "start": 0.5, "end": 0.9}}
                    ]
                },
                {
                    "name": "satisfaction",
                    "sets": [
                        {"name": "low", "function": {"kind": "left_shoulder", "start": 0.2, "end": 0.5}},
                        {"name": "high", "function": {"kind": "right_shoulder", "start": 0.6, "end": 0.9}}
                    ]
                }
            ],
            "rules": [
                {
                    "id": "authentic-novelty-fit",
                    "if": {
                        "op": "and",
                        "terms": [
                            {"op": "is", "variable": "authenticity", "set": "high"},
                            {"op": "is", "variable": "novelty", "set": "low"}
                        ]
                    },
                    "then": {"variable": "satisfaction", "set": "high"}
                },
                {
                    "id": "slow-service-penalty",
                    "if": {"op": "is", "variable": "wait_time", "set": "high"},
                    "then": {"variable": "satisfaction", "set": "low"}
                }
            ]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("satisfaction.high"));
    assert!(content.contains("authentic-novelty-fit"));
    assert!(content.contains("\"confidence\""));
}

#[tokio::test]
async fn classification_assigns_labels() {
    let result = run_with_input(
        ClassificationPack,
        serde_json::json!({
            "records": [[3.0, 3.0], [-3.0, -3.0], [2.0, 2.0], [-2.0, -2.0]],
            "weights": [1.0, 1.0],
            "bias": -2.0,
            "threshold": 0.5,
            "labels": ["spam", "not-spam"]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("spam"));
    assert!(content.contains("not-spam"));
    assert!(content.contains("probability"));
}

#[tokio::test]
async fn regression_predicts_values() {
    let result = run_with_input(
        RegressionPack,
        serde_json::json!({
            "records": [[1.0], [2.0], [3.0], [4.0]],
            "weights": [10.0],
            "bias": 5.0
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    // record [1.0] * 10.0 + 5.0 = 15.0
    assert!(content.contains("15"));
    // record [4.0] * 10.0 + 5.0 = 45.0
    assert!(content.contains("45"));
}

#[tokio::test]
async fn similarity_finds_nearest_pairs() {
    let result = run_with_input(
        SimilarityPack,
        serde_json::json!({
            "items": [
                {"id": "a", "features": [1.0, 0.0, 0.0]},
                {"id": "b", "features": [0.9, 0.1, 0.0]},
                {"id": "c", "features": [0.0, 0.0, 1.0]}
            ],
            "metric": "cosine",
            "top_k": 2
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("pairs"));
    // a and b should be most similar (both point roughly in same direction)
    let a_b = content
        .find("\"id_a\":\"a\"")
        .or_else(|| content.find("\"id_b\":\"a\""));
    assert!(a_b.is_some(), "pair involving 'a' should exist");
}

#[tokio::test]
async fn trend_detection_identifies_segments() {
    let result = run_with_input(
        TrendDetectionPack,
        serde_json::json!({
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "window": 3,
            "sensitivity": 1.0
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("segments"));
    assert!(content.contains("rising") || content.contains("falling"));
}

#[tokio::test]
async fn descriptive_stats_computes_summary() {
    let result = run_with_input(
        DescriptiveStatsPack,
        serde_json::json!({
            "values": [10.0, 20.0, 30.0, 40.0, 50.0],
            "percentiles": [25.0, 50.0, 75.0]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("mean"));
    assert!(content.contains("median"));
    assert!(content.contains("std_dev"));
    // mean of [10,20,30,40,50] = 30
    assert!(content.contains("30"));
}

#[tokio::test]
async fn sugeno_constant_rule_produces_output() {
    let result = run_with_input(
        SugenoInferencePack,
        serde_json::json!({
            "inputs": {"speed": 80.0},
            "variables": [
                {
                    "name": "speed",
                    "sets": [
                        {"name": "fast", "function": {"kind": "right_shoulder", "start": 50.0, "end": 100.0}}
                    ]
                }
            ],
            "rules": [
                {
                    "id": "speed-penalty",
                    "if": {"op": "is", "variable": "speed", "set": "fast"},
                    "then": {"kind": "constant", "value": 90.0}
                }
            ]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("output"));
    assert!(content.contains("speed-penalty"));
    assert!(content.contains("\"confidence\""));
}

#[tokio::test]
async fn sugeno_linear_rule_produces_output() {
    let result = run_with_input(
        SugenoInferencePack,
        serde_json::json!({
            "inputs": {"x": 2.0, "y": 3.0},
            "variables": [
                {
                    "name": "x",
                    "sets": [{"name": "pos", "function": {"kind": "right_shoulder", "start": 0.0, "end": 4.0}}]
                },
                {
                    "name": "y",
                    "sets": [{"name": "pos", "function": {"kind": "right_shoulder", "start": 0.0, "end": 6.0}}]
                }
            ],
            "rules": [
                {
                    "if": {"op": "and", "terms": [
                        {"op": "is", "variable": "x", "set": "pos"},
                        {"op": "is", "variable": "y", "set": "pos"}
                    ]},
                    "then": {"kind": "linear", "intercept": 1.0, "coefficients": {"x": 2.0, "y": 3.0}}
                }
            ]
        }),
    )
    .await;

    assert!(result.converged);
    let content = fact_content(&result.context.get(ContextKey::Strategies)[0]);
    assert!(content.contains("output"));
}

#[tokio::test]
async fn tsukamoto_inference_produces_crisp_output() {
    let result = run_with_input(
        TsukamotoInferencePack,
        serde_json::json!({
            "inputs": {"quality": 0.6},
            "variables": [
                {
                    "name": "quality",
                    "sets": [
                        {"name": "good", "function": {"kind": "right_shoulder", "start": 0.3, "end": 0.9}}
                    ]
                },
                {
                    "name": "score",
                    "sets": [
                        {"name": "high", "function": {"kind": "right_shoulder", "start": 0.0, "end": 1.0}}
                    ]
                }
            ],
            "rules": [
                {
                    "id": "quality-to-score",
                    "if": {"op": "is", "variable": "quality", "set": "good"},
                    "then": {"variable": "score", "set": "high"}
                }
            ]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("output"));
    assert!(content.contains("quality-to-score"));
    assert!(content.contains("\"confidence\""));
}

#[tokio::test]
async fn naive_bayes_classifies_point() {
    let result = run_with_input(
        NaiveBayesPack,
        serde_json::json!({
            "classes": [
                {
                    "name": "spam",
                    "prior": 0.4,
                    "feature_params": [
                        {"mean": 10.0, "std_dev": 2.0},
                        {"mean": 5.0, "std_dev": 1.0}
                    ]
                },
                {
                    "name": "not-spam",
                    "prior": 0.6,
                    "feature_params": [
                        {"mean": 2.0, "std_dev": 1.0},
                        {"mean": 8.0, "std_dev": 2.0}
                    ]
                }
            ],
            "features": [9.0, 5.5]
        }),
    )
    .await;

    assert!(result.converged);
    let strategies = result.context.get(ContextKey::Strategies);
    assert_eq!(strategies.len(), 1);
    let content = fact_content(&strategies[0]);
    assert!(content.contains("predicted"));
    assert!(content.contains("spam"));
    assert!(content.contains("probabilities"));
    assert!(content.contains("\"confidence\""));
}

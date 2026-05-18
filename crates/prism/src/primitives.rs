use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
pub use std::num::NonZeroUsize;

/// A smoothing or probability factor in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct UnitFraction(f64);

impl UnitFraction {
    pub fn new(v: f64) -> Result<Self, String> {
        if (0.0..=1.0).contains(&v) {
            Ok(Self(v))
        } else {
            Err(format!("{v} is outside [0.0, 1.0]"))
        }
    }
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Serialize for UnitFraction {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(self.0)
    }
}

struct UnitFractionVisitor;

impl Visitor<'_> for UnitFractionVisitor {
    type Value = UnitFraction;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a float in [0.0, 1.0]")
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<UnitFraction, E> {
        UnitFraction::new(v).map_err(de::Error::custom)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<UnitFraction, E> {
        self.visit_f64(v as f64)
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<UnitFraction, E> {
        self.visit_f64(v as f64)
    }
}

impl<'de> Deserialize<'de> for UnitFraction {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_f64(UnitFractionVisitor)
    }
}

/// A positive z-score threshold (> 0.0).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ZScoreThreshold(f64);

impl ZScoreThreshold {
    pub fn new(v: f64) -> Result<Self, String> {
        if v > 0.0 {
            Ok(Self(v))
        } else {
            Err(format!("{v} must be > 0.0"))
        }
    }
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Serialize for ZScoreThreshold {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(self.0)
    }
}

struct ZScoreThresholdVisitor;

impl Visitor<'_> for ZScoreThresholdVisitor {
    type Value = ZScoreThreshold;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a float > 0.0")
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<ZScoreThreshold, E> {
        ZScoreThreshold::new(v).map_err(de::Error::custom)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<ZScoreThreshold, E> {
        self.visit_f64(v as f64)
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<ZScoreThreshold, E> {
        self.visit_f64(v as f64)
    }
}

impl<'de> Deserialize<'de> for ZScoreThreshold {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_f64(ZScoreThresholdVisitor)
    }
}

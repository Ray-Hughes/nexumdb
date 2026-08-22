//! Property values for node metadata and edge properties.
//!
//! This exists because `serde_json::Value` cannot be encoded by a non
//! self-describing format: its `Deserialize` calls `deserialize_any`, which
//! bincode rejects. Rather than give up JSON-shaped properties, [`PropertyValue`]
//! carries its own serde impls that branch on `is_human_readable()` — natural
//! JSON on the wire, an explicitly tagged encoding on disk. Callers see plain
//! JSON in both directions and never have to think about it.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A JSON-shaped value.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum PropertyValue {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    List(Vec<PropertyValue>),
    Map(BTreeMap<String, PropertyValue>),
}

/// A bag of properties.
pub type Properties = BTreeMap<String, PropertyValue>;

impl PropertyValue {
    /// Type name, for error messages and the viewer's property inspector.
    pub fn type_name(&self) -> &'static str {
        match self {
            PropertyValue::Null => "null",
            PropertyValue::Bool(_) => "bool",
            PropertyValue::Int(_) => "int",
            PropertyValue::Float(_) => "float",
            PropertyValue::Text(_) => "text",
            PropertyValue::List(_) => "list",
            PropertyValue::Map(_) => "map",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropertyValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Numeric view, widening ints to floats so `> 3` works against both.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropertyValue::Int(i) => Some(*i as f64),
            PropertyValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PropertyValue::Int(i) => Some(*i),
            // Only when it is genuinely integral — silently truncating 1.9 to
            // 1 in a filter would be a quiet wrong answer.
            PropertyValue::Float(f) if f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[PropertyValue]> {
        match self {
            PropertyValue::List(l) => Some(l),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, PropertyValue::Null)
    }

    /// Render for terminal display.
    pub fn to_display_string(&self) -> String {
        match self {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::Text(s) => s.clone(),
            PropertyValue::List(_) | PropertyValue::Map(_) => {
                serde_json::to_string(self).unwrap_or_else(|_| "<unrenderable>".into())
            }
        }
    }
}

// Ergonomic constructors, so callers write `edge.with("score", 0.82)`.
impl From<bool> for PropertyValue {
    fn from(v: bool) -> Self {
        PropertyValue::Bool(v)
    }
}
impl From<i64> for PropertyValue {
    fn from(v: i64) -> Self {
        PropertyValue::Int(v)
    }
}
impl From<i32> for PropertyValue {
    fn from(v: i32) -> Self {
        PropertyValue::Int(v as i64)
    }
}
impl From<u32> for PropertyValue {
    fn from(v: u32) -> Self {
        PropertyValue::Int(v as i64)
    }
}
impl From<usize> for PropertyValue {
    fn from(v: usize) -> Self {
        PropertyValue::Int(v as i64)
    }
}
impl From<f64> for PropertyValue {
    fn from(v: f64) -> Self {
        PropertyValue::Float(v)
    }
}
impl From<f32> for PropertyValue {
    fn from(v: f32) -> Self {
        PropertyValue::Float(v as f64)
    }
}
impl From<String> for PropertyValue {
    fn from(v: String) -> Self {
        PropertyValue::Text(v)
    }
}
impl From<&str> for PropertyValue {
    fn from(v: &str) -> Self {
        PropertyValue::Text(v.to_string())
    }
}
impl<T: Into<PropertyValue>> From<Option<T>> for PropertyValue {
    fn from(v: Option<T>) -> Self {
        v.map_or(PropertyValue::Null, Into::into)
    }
}
impl<T: Into<PropertyValue>> From<Vec<T>> for PropertyValue {
    fn from(v: Vec<T>) -> Self {
        PropertyValue::List(v.into_iter().map(Into::into).collect())
    }
}

impl From<serde_json::Value> for PropertyValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => PropertyValue::Null,
            serde_json::Value::Bool(b) => PropertyValue::Bool(b),
            serde_json::Value::Number(n) => match n.as_i64() {
                Some(i) => PropertyValue::Int(i),
                None => PropertyValue::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(s) => PropertyValue::Text(s),
            serde_json::Value::Array(a) => {
                PropertyValue::List(a.into_iter().map(PropertyValue::from).collect())
            }
            serde_json::Value::Object(o) => PropertyValue::Map(
                o.into_iter()
                    .map(|(k, v)| (k, PropertyValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<PropertyValue> for serde_json::Value {
    fn from(value: PropertyValue) -> Self {
        match value {
            PropertyValue::Null => serde_json::Value::Null,
            PropertyValue::Bool(b) => serde_json::Value::Bool(b),
            PropertyValue::Int(i) => serde_json::Value::Number(i.into()),
            PropertyValue::Float(f) => serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                // JSON has no NaN or infinity; null is the honest mapping.
                .unwrap_or(serde_json::Value::Null),
            PropertyValue::Text(s) => serde_json::Value::String(s),
            PropertyValue::List(l) => {
                serde_json::Value::Array(l.into_iter().map(Into::into).collect())
            }
            PropertyValue::Map(m) => serde_json::Value::Object(
                m.into_iter().map(|(k, v)| (k, v.into())).collect(),
            ),
        }
    }
}

/// Borrowed mirror used for the binary encoding.
#[derive(Serialize)]
enum TaggedRef<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(&'a str),
    List(&'a [PropertyValue]),
    Map(&'a BTreeMap<String, PropertyValue>),
}

/// Owned mirror used for the binary decoding.
#[derive(Deserialize)]
enum Tagged {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    List(Vec<PropertyValue>),
    Map(BTreeMap<String, PropertyValue>),
}

impl From<Tagged> for PropertyValue {
    fn from(t: Tagged) -> Self {
        match t {
            Tagged::Null => PropertyValue::Null,
            Tagged::Bool(b) => PropertyValue::Bool(b),
            Tagged::Int(i) => PropertyValue::Int(i),
            Tagged::Float(f) => PropertyValue::Float(f),
            Tagged::Text(s) => PropertyValue::Text(s),
            Tagged::List(l) => PropertyValue::List(l),
            Tagged::Map(m) => PropertyValue::Map(m),
        }
    }
}

impl Serialize for PropertyValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !serializer.is_human_readable() {
            let tagged = match self {
                PropertyValue::Null => TaggedRef::Null,
                PropertyValue::Bool(b) => TaggedRef::Bool(*b),
                PropertyValue::Int(i) => TaggedRef::Int(*i),
                PropertyValue::Float(f) => TaggedRef::Float(*f),
                PropertyValue::Text(s) => TaggedRef::Text(s),
                PropertyValue::List(l) => TaggedRef::List(l),
                PropertyValue::Map(m) => TaggedRef::Map(m),
            };
            return tagged.serialize(serializer);
        }

        match self {
            PropertyValue::Null => serializer.serialize_none(),
            PropertyValue::Bool(b) => serializer.serialize_bool(*b),
            PropertyValue::Int(i) => serializer.serialize_i64(*i),
            PropertyValue::Float(f) => serializer.serialize_f64(*f),
            PropertyValue::Text(s) => serializer.serialize_str(s),
            PropertyValue::List(l) => l.serialize(serializer),
            PropertyValue::Map(m) => m.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PropertyValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return Tagged::deserialize(deserializer).map(PropertyValue::from);
        }
        deserializer.deserialize_any(PropertyVisitor)
    }
}

struct PropertyVisitor;

impl<'de> Visitor<'de> for PropertyVisitor {
    type Value = PropertyValue;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(PropertyValue::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(PropertyValue::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        PropertyValue::deserialize(d)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(PropertyValue::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(PropertyValue::Int(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        // Values beyond i64 keep their magnitude as a float rather than
        // wrapping into a negative number.
        Ok(i64::try_from(v).map_or(PropertyValue::Float(v as f64), PropertyValue::Int))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(PropertyValue::Float(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(PropertyValue::Text(v.to_string()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(PropertyValue::Text(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(item) = seq.next_element()? {
            out.push(item);
        }
        Ok(PropertyValue::List(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut out = BTreeMap::new();
        while let Some((key, value)) = map.next_entry()? {
            out.insert(key, value);
        }
        Ok(PropertyValue::Map(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PropertyValue {
        PropertyValue::Map(BTreeMap::from([
            ("text".into(), PropertyValue::Text("hello".into())),
            ("int".into(), PropertyValue::Int(-42)),
            ("float".into(), PropertyValue::Float(1.5)),
            ("bool".into(), PropertyValue::Bool(true)),
            ("null".into(), PropertyValue::Null),
            (
                "list".into(),
                PropertyValue::List(vec![PropertyValue::Int(1), PropertyValue::Text("two".into())]),
            ),
        ]))
    }

    #[test]
    fn json_representation_is_natural() {
        let json = serde_json::to_string(&PropertyValue::Text("hi".into())).unwrap();
        assert_eq!(json, r#""hi""#);
        assert_eq!(serde_json::to_string(&PropertyValue::Int(7)).unwrap(), "7");
        assert_eq!(serde_json::to_string(&PropertyValue::Null).unwrap(), "null");
        assert_eq!(
            serde_json::to_string(&PropertyValue::List(vec![PropertyValue::Int(1)])).unwrap(),
            "[1]"
        );
    }

    #[test]
    fn roundtrips_through_json() {
        let value = sample();
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(value, serde_json::from_str::<PropertyValue>(&json).unwrap());
    }

    #[test]
    fn roundtrips_through_bincode() {
        let value = sample();
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&value, cfg).unwrap();
        let (decoded, _): (PropertyValue, _) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn converts_to_and_from_serde_json_value() {
        let json: serde_json::Value = serde_json::json!({
            "a": 1, "b": [true, null, "x"], "c": {"d": 2.5}
        });
        let property = PropertyValue::from(json.clone());
        assert_eq!(serde_json::Value::from(property), json);
    }

    #[test]
    fn numeric_accessors_widen_but_do_not_truncate() {
        assert_eq!(PropertyValue::Int(3).as_f64(), Some(3.0));
        assert_eq!(PropertyValue::Float(3.0).as_i64(), Some(3));
        assert_eq!(PropertyValue::Float(3.9).as_i64(), None);
        assert_eq!(PropertyValue::Text("3".into()).as_f64(), None);
    }

    #[test]
    fn non_finite_floats_degrade_to_json_null() {
        assert_eq!(
            serde_json::Value::from(PropertyValue::Float(f64::NAN)),
            serde_json::Value::Null
        );
    }

    #[test]
    fn huge_unsigned_integers_keep_their_magnitude() {
        let value: PropertyValue = serde_json::from_str("18446744073709551615").unwrap();
        assert!(value.as_f64().unwrap() > 1e19);
    }
}

"""Validated data contracts shared across Viralyst modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ObjectStats(BaseModel):
    counts: dict[str, int]
    unique_classes: int
    avg_objects_per_frame: float
    dominant_class: str | None


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    no_speech_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "TranscriptSegment":
        if self.end < self.start:
            raise ValueError("segment end must be greater than or equal to start")
        return self


class TranscriptResult(BaseModel):
    transcript: str
    language: str
    segments: list[TranscriptSegment]
    speech_duration_sec: float = Field(ge=0.0)
    word_count: int = Field(ge=0)


class VideoFeatures(BaseModel):
    video_hash: str
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    sampled_frame_count: int
    scene_changes: int
    scene_change_rate: float
    avg_brightness: float
    brightness_variance: float
    motion_score: float
    aspect_ratio: float
    is_vertical: bool
    objects: ObjectStats
    transcript: str
    language: str
    word_count: int
    speech_duration_sec: float
    speech_rate_wps: float
    silence_ratio: float
    has_speech: bool
    hook_transcript: str
    pacing_label: Literal["slow", "moderate", "fast"]
    density_label: Literal["sparse", "balanced", "busy"]


class Persona(BaseModel):
    id: int
    name: str
    age: int = Field(ge=16, le=55)
    profession: str
    archetype: str
    interests: list[str]
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    attention_span: float = Field(ge=0.0, le=1.0)
    skepticism: float = Field(ge=0.0, le=1.0)
    share_propensity: float = Field(ge=0.0, le=1.0)
    comment_propensity: float = Field(ge=0.0, le=1.0)
    daily_scroll_hours: float = Field(ge=0.5, le=6.0)


class Reaction(BaseModel):
    persona_id: int
    wave: int
    watch_percentage: int = Field(ge=0, le=100)
    skipped: bool
    liked: bool
    commented: bool
    shared: bool
    followed_creator: bool
    purchase_intent: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=280)
    latency_ms: int = 0
    fallback: bool = False

    @model_validator(mode="after")
    def enforce_consistency(self) -> "Reaction":
        if self.skipped:
            self.watch_percentage = min(self.watch_percentage, 30)
            self.liked = False
            self.commented = False
            self.shared = False
            self.followed_creator = False
        if self.watch_percentage < 25:
            self.shared = False
            self.followed_creator = False
        if self.followed_creator and not (self.liked or self.shared):
            self.liked = True
        return self


class Wave(BaseModel):
    index: int
    exposed_ids: list[int]
    new_exposures: int
    cumulative_reached: int
    shares: int
    comments: int
    likes: int
    skips: int
    avg_watch: float
    continued: bool


class SegmentMetrics(BaseModel):
    segment: str
    n: int
    avg_watch: float
    like_rate: float
    share_rate: float
    comment_rate: float
    skip_rate: float


class EngagementMetrics(BaseModel):
    n_reactions: int
    completion_rate: float
    skip_rate: float
    like_rate: float
    share_rate: float
    comment_rate: float
    follow_rate: float
    avg_purchase_intent: float
    reach: int
    reach_pct: float
    virality_score: float
    segments: list[SegmentMetrics]


class RecommendationItem(BaseModel):
    problem: str
    likely_cause: str
    recommendation: str
    priority: Literal["high", "medium", "low"]
    target_segment: str | None


class Recommendations(BaseModel):
    summary: str
    strengths: list[str]
    items: list[RecommendationItem]

__all__ = ["SpeakerDetectionPipeline", "TwoPersonContourProcessor", "run_single_image_mode"]


def __getattr__(name):
    if name == "SpeakerDetectionPipeline":
        from .speaker_pipeline import SpeakerDetectionPipeline

        return SpeakerDetectionPipeline
    if name in {"TwoPersonContourProcessor", "run_single_image_mode"}:
        from .two_person_contour_mode import TwoPersonContourProcessor, run_single_image_mode

        return {"TwoPersonContourProcessor": TwoPersonContourProcessor, "run_single_image_mode": run_single_image_mode}[name]
    raise AttributeError(name)

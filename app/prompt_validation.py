class PromptLimitError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        scene_id: str,
        prompt_type: str,
        actual_length: int,
        limit: int,
        length_unit: str,
        required_fields: tuple[str, ...],
    ) -> None:
        self.scene_id = scene_id
        self.prompt_type = prompt_type
        self.actual_length = actual_length
        self.limit = limit
        self.length_unit = length_unit
        self.required_fields = required_fields
        fields = ", ".join(required_fields) or "not determined"
        super().__init__(
            f"{scene_id} {prompt_type}: {message} "
            f"Actual prompt length: {actual_length} "
            f"{length_unit}; limit: {limit}. Required fields contributing "
            f"to overflow: {fields}."
        )

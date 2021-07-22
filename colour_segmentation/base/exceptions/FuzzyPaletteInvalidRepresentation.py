class FuzzyPaletteInvalidRepresentation(Exception):
    """
    An Exception indicating that not enough classes have been provided to represent the fuzzy sets.
    """

    def __init__(self, provided_labels: int, needed_labels: int):
        super().__init__(f"{needed_labels} labels are needed to represent the selected method. "
                         f"Only {provided_labels} are provided.")

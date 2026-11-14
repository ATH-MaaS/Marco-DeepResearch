class PlacementGroup:
    def __init__(self, *args, **kwargs):
        self.id = "local-pg"


def placement_group(*args, **kwargs):
    return PlacementGroup()

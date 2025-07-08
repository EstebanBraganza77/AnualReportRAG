from .documents import DMartARExtractor, MRFExtractor

extractors = {
    "DMART": DMartARExtractor,
    "MRF": MRFExtractor,
    "TATAMOTORS": MRFExtractor
}
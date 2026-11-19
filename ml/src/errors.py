"""Error khusus (tanpa dependensi torch agar ringan diimpor backend)."""


class ImageDecodeError(ValueError):
    """File bukan gambar yang valid."""


class ModelNotLoadedError(FileNotFoundError):
    """Bobot model tidak ditemukan."""


class InvalidInputError(ValueError):
    """Input data atau format citra tidak valid."""

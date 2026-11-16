"""Error khusus (tanpa dependensi torch agar ringan diimpor backend)."""


class ImageDecodeError(ValueError):
    """File bukan gambar yang valid."""

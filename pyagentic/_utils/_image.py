from PIL.Image import Image
import base64
import io


def _encode_image(image: Image) -> str:
    """
    Convert a PIL Image to a base64-encoded data URL.

    Args:
        image: PIL Image object to encode

    Returns:
        Base64-encoded data URL string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"

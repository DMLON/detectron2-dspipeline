from fastapi import APIRouter
from ..licenseplate_detector import LicenseplateDetector
from starlette.responses import JSONResponse

router = APIRouter()


@router.post('/detect_licenseplate')
def extract_name(input_image: dict):
    licenseplate_detector = LicenseplateDetector()
    return JSONResponse(licenseplate_detector.detect(input_image))

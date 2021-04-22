from fastapi import FastAPI
from .router import licenseplate_detector_router

app = FastAPI()
app.include_router(licenseplate_detector_router.router, prefix='/anpr')


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Licenseplate detector is ready!'

from pydantic import BaseModel

class hepsiburada(BaseModel):
    memory: float
    ram: float
    screen_size: float
    power:float
    front_camera:float
    rc1:float
    rc3:float
    rc5:float
    rc7:float

    class Config:
        schema_extra = {
            "example": {
                "memory": 128.0,
                "ram": 8.0,
                "screen_size": 6.40,
                "power": 4310.0,
                "front_camera": 32.0,
                "rc1": 48.0,
                "rc3": 8.0 ,
                "rc5": 2.0,
                "rc7": 2.0,

            }
        }

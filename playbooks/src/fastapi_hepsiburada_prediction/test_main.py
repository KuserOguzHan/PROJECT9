from fastapi.testclient import TestClient

try:
    from main import app
except:

    from fastapi_hepsiburada_prediction.main import app

client = TestClient(app)


def test_predict_hepsiburada():
    response = client.post("/prediction/hepsiburada", json={
        "memory": 128.0,
        "ram": 8.0,
        "screen_size": 6.40,
        "power": 4310.0,
        "front_camera": 32.0,
        "rc1": 48.0,
        "rc3": 8.0,
        "rc5": 2.0,
        "rc7": 2.0
    })

    assert response.status_code == 200
    assert isinstance(response.json()['result'], float), 'Result wrong type!'

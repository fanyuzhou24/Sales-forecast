from sales_forecast_system.app import create_app


def test_app_create() -> None:
    app = create_app()
    assert app.title == "Sales Forecast System"

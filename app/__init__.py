import math

from flask import Flask

from .routes import bp
from .utils import filer_url, format_large_number


def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)

    @app.template_filter('currency')
    def currency_filter(value):
        if value is None:
            return '-'
        return f"${value:,.2f}"

    @app.template_filter()
    def format_percent_or_na(value, decimals=2, times_100=False):
        try:
            fval = float(value)
            if math.isnan(fval):
                return "N/A"
            if times_100:
                fval = fval * 100
            return f"{fval:,.{decimals}f}%"
        except (ValueError, TypeError):
            return "N/A"

    @app.template_filter()
    def format_number_or_na(value, decimals=2):
        try:
            fval = float(value)
            if math.isnan(fval):
                return "N/A"

            format_str = f"{{:,.{decimals}f}}".format(fval)
            return format_str
        except (ValueError, TypeError):
            return "N/A"

    app.jinja_env.globals.update(filer_url=filer_url)
    app.jinja_env.filters['format_large_number'] = format_large_number

    return app

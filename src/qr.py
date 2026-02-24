import iws
import vtb_qr as vq

from config import CALYPSO_PASSWORD, CALYPSO_LOGIN

vq.load_xll_and_static()


def get_calypso_session(url='IWSCPSLIVE', port=8200):
    session = iws.CalypsoSession(url, port, '/CalypsoIntegration/avro')
    session.login(CALYPSO_LOGIN, CALYPSO_PASSWORD)
    return session


def get_mos_dates(sdate, edate, *, incl_end):
    dates = vq.DateScheduler("MOS", vq.toVariant(sdate), "1bd", vq.toVariant(edate), "F")
    dates = [date.date() for date in vq.toNumpy(dates) if (date.date() <= edate if incl_end else date.date() < edate)]
    return dates


def get_mos_dates_range(edate, nb_days):
    assert nb_days > 0
    sdate = shift_mos_date(edate, -nb_days)
    days_ = get_mos_dates(sdate, edate, incl_end=True)
    assert len(days_) >= nb_days
    return days_[-nb_days:]


def shift_mos_date(date, shift_bdays):
    res = vq.DateAdd("MOS", vq.toVariant(date), vq.toVariant(f'{shift_bdays}BD'), "Actual")
    return vq.toNumpy(res).date()


def is_mos_business_date(date):
    return vq.DateIsBusiness("MOS", vq.toVariant(date))

from dataclasses import dataclass

import pandas as pd


@dataclass
class Id:
    secid: str
    isin: str


_OFZ_RECORDS = [{'secid': 'SU26238RMFS4', 'isin': 'RU000A1038V6'},
                {'secid': 'SU26230RMFS1', 'isin': 'RU000A100EF5'},
                {'secid': 'SU26248RMFS3', 'isin': 'RU000A108EH4'},
                {'secid': 'SU26243RMFS4', 'isin': 'RU000A106E90'},
                {'secid': 'SU26240RMFS0', 'isin': 'RU000A103BR0'},
                {'secid': 'SU26254RMFS1', 'isin': 'RU000A10D533'},
                {'secid': 'SU26247RMFS5', 'isin': 'RU000A108EF8'},
                {'secid': 'SU26233RMFS5', 'isin': 'RU000A101F94'},
                {'secid': 'SU26253RMFS3', 'isin': 'RU000A10D517'},
                {'secid': 'SU26250RMFS9', 'isin': 'RU000A10BVH7'},
                {'secid': 'SU26225RMFS1', 'isin': 'RU000A0ZYUB7'},
                {'secid': 'SU26246RMFS7', 'isin': 'RU000A108EE1'},
                {'secid': 'SU26245RMFS9', 'isin': 'RU000A108EG6'},
                {'secid': 'SU26221RMFS0', 'isin': 'RU000A0JXFM1'},
                {'secid': 'SU26244RMFS2', 'isin': 'RU000A1074G2'},
                {'secid': 'SU26252RMFS5', 'isin': 'RU000A10D4Y2'},
                {'secid': 'SU26241RMFS8', 'isin': 'RU000A105FZ9'},
                {'secid': 'SU26249RMFS1', 'isin': 'RU000A10BVC8'},
                {'secid': 'SU26239RMFS2', 'isin': 'RU000A103901'},
                {'secid': 'SU26218RMFS6', 'isin': 'RU000A0JVW48'},
                {'secid': 'SU26235RMFS0', 'isin': 'RU000A1028E3'},
                {'secid': 'SU26251RMFS7', 'isin': 'RU000A10CKT3'},
                {'secid': 'SU26228RMFS5', 'isin': 'RU000A100A82'},
                {'secid': 'SU26242RMFS6', 'isin': 'RU000A105RV3'},
                {'secid': 'SU26224RMFS4', 'isin': 'RU000A0ZYUA9'},
                {'secid': 'SU26237RMFS6', 'isin': 'RU000A1038Z7'},
                {'secid': 'SU26236RMFS8', 'isin': 'RU000A102BT8'},
                {'secid': 'SU26212RMFS9', 'isin': 'RU000A0JTK38'},
                {'secid': 'SU26232RMFS7', 'isin': 'RU000A1014N4'},
                {'secid': 'SU26207RMFS9', 'isin': 'RU000A0JS3W6'},
                {'secid': 'SU26226RMFS9', 'isin': 'RU000A0ZZYW2'},
                {'secid': 'SU26219RMFS4', 'isin': 'RU000A0JWM07'},
                {'secid': 'SU26229RMFS3', 'isin': 'RU000A100EG3'},
                {'secid': 'SU26234RMFS3', 'isin': 'RU000A101QE0'},
                {'secid': 'SU26222RMFS8', 'isin': 'RU000A0JXQF2'},
                {'secid': 'SU26227RMFS7', 'isin': 'RU000A1007F4'},
                {'secid': 'SU26223RMFS6', 'isin': 'RU000A0ZYU88'},
                {'secid': 'SU25084RMFS3', 'isin': 'RU000A101FA1'}]

OFZ = {int(d['secid'][2:7]): Id(d['secid'], d['isin']) for d in _OFZ_RECORDS}

def symbol_to_series(s: pd.Series) -> pd.Series:
    return s.str.slice(2, 7).astype(int)

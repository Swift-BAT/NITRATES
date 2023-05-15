import requests
import json
from datetime import datetime
import pandas as pd
import numpy as np

BASE_URL = "https://guano.swift.psu.edu/api/"

class API:
    def __init__(self, api_token, base_url=BASE_URL):
        self.base_url = base_url
        self.api_token = api_token

    def _send_request(self, method, endpoint, payload=None):
        url = f"{self.base_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}

        if payload is None:
            payload = {}
        elif isinstance(payload, str):
            payload = json.loads(payload)

        payload['api_token'] = self.api_token
        if method == 'GET':
            response = requests.get(url = url, json = payload)
        else:
            response = requests.request(method, url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code}")

    def _validate_trigger_type(self, trigger_type):
        valid_trigger_types = ['GRB', 'GW', 'neutrino', 'FRB']
        return trigger_type in valid_trigger_types
    
    def _validate_log_type(self,args):
        column_names = [
                "EvDataFound",
                "AllDataFound",
                "NITRATESstart",
                "BkgStart",
                "SplitRatesStart",
                "SplitRatesDone",
                "SquareSeeds",
                "TimeBins",
                "TotalSeeds",
                "IFOVStart",
                "IFOVjobs",
                "IFOVDone",
                "OFOVStart",
                "OFOVjobs",
                "OFOVDone",
                "IFOVfilesTot",
                "IFOVfilesDone",
                "OFOVfilesTot",
                "OFOVfilesDone"]
        
        return any(arg in column_names for arg in args)

    def _check_datetime_format(self, datetime_str):
        try:
            datetime.fromisoformat(datetime_str)
            return True
        except ValueError:
            return False

    def get_trigs(self):
        response = self._send_request("GET", "trigs")

        return response

    def get_trig(self, trigger_time):

        if not self._check_datetime_format(trigger_time):
            raise ValueError("trigger_time must be in the format 'YYYY-MM-DDTHH:MM:SS+00:00'.")
        
        payload = {
            'trigger_time': trigger_time
        }

        response = self._send_request("GET", "trig", payload)

        return response

    def post_trig(self, trigger_time, trigger_instrument, trigger_type, trigger_name, mou=False,
                  ra=None,dec=None,pos_error=None):
        if not self._validate_trigger_type(trigger_type):
            raise ValueError(f"Invalid trigger_type. Must be one of {valid_trigger_types}")
        if not self._check_datetime_format(trigger_time):
            raise ValueError("trigger_time must be in the format 'YYYY-MM-DDTHH:MM:SS+00:00'.")
        if not isinstance(trigger_instrument, str):
            raise ValueError("trigger_instrument must be a string.")
        if not isinstance(trigger_type, str):
            raise ValueError("trigger_type must be a string.")
        if not isinstance(trigger_name, str):
            raise ValueError("trigger_name must be a string.")
        if not isinstance(mou, bool):
            raise ValueError("mou must be a boolean.")

        payload = {
            'trigger_time': trigger_time,
            'trigger_instrument': trigger_instrument,
            'trigger_type': trigger_type,
            'trigger_name': trigger_name,
            'MOU': mou
        }

        if ra and dec and pos_error:
            if not isinstance(ra, float) or not isinstance(dec, float) or not isinstance(pos_error, float):
                raise ValueError("ra,dec,pos_error must all be floats in degrees.")  
            payload['ra'] = ra
            payload['dec'] = dec
            payload['pos_error'] = pos_error

        response = self._send_request("POST", "trigs", payload)

        return response

    def claim(self, claim_data):
        response = self._send_request("POST", "claim", payload)
        return response

    def post_log(self, **kwargs):
        if 'trigger' not in kwargs or 'config_id' not in kwargs:
            raise ValueError("trigger and config_id are required.")
        if not self._validate_log_type(kwargs):
            raise ValueError("No valid log type.")
        
        payload = kwargs
        
        response = self._send_request("POST", "logs", payload)
        return response

    def post_nitrates_results(self, trigger, config_id, result_type, result_data):
        valid_result_types = ['n_FULLRATE', 'n_SPLITRATE', 'n_OUTFOV', 'n_INFOV', 'n_TOP']

        if result_type not in valid_result_types:
            raise ValueError(f"Invalid result_type. Must be one of {valid_result_types}")
        if not isinstance(result_data, pd.DataFrame):
            raise ValueError("result_data must be a pandas dataframe.")
        if not len(np.unique(result_data['trigger_id']))==1:
            raise ValueError("result_data must only contain results for one trigger.")
        if not len(np.unique(result_data['config_id']))==1:
            raise ValueError("result_data must only contain results for one config.")
        if result_data['trigger_id'].iloc[0] != trigger:
            raise ValueError("result_data must only contain results for the specified trigger.")

        result_data = result_data.to_json()

        payload = {
            'trigger': trigger,
            'config_id': config_id,
            'result_type': result_type,
            'result': result_data 
        }

        response = self._send_request("POST", "nitrates_results", payload)
        return response

    def get_nitrates_results(self):
        url = self.base_url + '/nitrates_results'
        response = requests.get(url)
        return response.json()

    # Add other methods for the remaining endpoints

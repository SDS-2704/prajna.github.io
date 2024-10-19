# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import datetime as dt
import string
import random
from typing import Any, Text, Dict, List
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import platform
import sqlalchemy
from sqlalchemy import create_engine
from idsp_connection_info import *
import psycopg2
SQL_CONN = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
engine = create_engine(SQL_CONN)
schema = "incedo_fs_schema"
#with engine.connect() as conn:
#        raw_accounts_df = pd.read_sql("""SELECT * FROM "incedo_fs_schema"."autoloans_accounts_data";""".format(schema), conn)
        
# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#

class GetAllContextsInLighthouse(Action):
    def name(self) -> Text:
        return "action_talk_to_lighthouse_get_all_contexts"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#       filter_ = '{"Month":"Apr-21","DaysPastDue":"30-59"}'
        contexts_list = pd.read_sql("""SELECT schema_name, SUBSTRING(schema_name, POSITION('_' IN schema_name)+1, POSITION('schema' IN schema_name) - (POSITION('_' IN schema_name)+2)) AS CONTEXT FROM information_schema.schemata where schema_name like 'incedo%%schema';""", engine)            
        contexts_list = contexts_list['context'].values.tolist()
        dispatcher.utter_message(text=f"Which context do you want to login to? \nWe have these contexts active currently in Lighthouse:-\n{contexts_list}.\n\nJust type - Log in to xyz context. Here, xyz is your context of interest, mentioned in the list above.")
        return []

class GetAllProductFamiliesInThatContext(Action):
    def name(self) -> Text:
        return "action_talk_to_lighthouse_get_all_product_families_in_context"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        lighthouse_context_ = tracker.get_slot("lighthouse_context")
#            query
#           filter_ = '{"Month":"Apr-21","DaysPastDue":"30-59"}'
        product_families_list = pd.read_sql("""SELECT distinct asset_class_id FROM "incedo_{}_schema"."asset_class_master";""".format(lighthouse_context_), engine)      
        product_families_list = product_families_list.values.tolist()    
#           raw_accounts_df.set_index('acc_id', inplace=True)
        dispatcher.utter_message(text=f"Which product family do you want me to focus on? We currently have these product families existing on pedemo context : {product_families_list}.")
        return []


class GetAllKPITreesInThatProductFamily(Action):
    def name(self) -> Text:
        return "action_talk_to_lighthouse_get_all_kpitrees_in_productfamily_of_that_context"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        lighthouse_context_ = tracker.get_slot("lighthouse_context")
        product_family_ = tracker.get_slot("product_family")
#            query
#       filter_ = '{"Month":"Apr-21","DaysPastDue":"30-59"}'
        kpitrees_list = pd.read_sql("""SELECT distinct CAST(kpi_tree_id AS TEXT ) || ' - ' || kpi_tree_name AS kpi_tree_name FROM "incedo_{}_schema"."kpi_tree_master" where kpi_tree_domain = '{}' AND kpi_tree_status = 'Published';""".format(lighthouse_context_,product_family_), engine)            
        kpitrees_list = kpitrees_list.values.tolist()
#       raw_accounts_df.set_index('acc_id', inplace=True)
#       dispatcher.utter_message(text=f"{product_family_}")
        dispatcher.utter_message(text=f"Which KPI Tree to check? You can type its full name. Here's the list of published KPI Trees identified within {product_family_} :-\n {kpitrees_list}.")
        return []

class GetAllNodesInThatKPITree(Action):
    def name(self) -> Text:
        return "action_talk_to_lighthouse_get_all_nodes_in_that_kpi_tree"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        lighthouse_context_ = tracker.get_slot("lighthouse_context")
        product_family_ = tracker.get_slot("product_family")
        kpi_tree_id_ = tracker.get_slot("kpi_tree_id")
        kpi_tree_name_ = tracker.get_slot("kpi_tree_name")
#            query
#       filter_ = '{"Month":"Apr-21","DaysPastDue":"30-59"}'
        if kpi_tree_id_ is not None:
           nodes_list = pd.read_sql("""SELECT node_name FROM "incedo_{}_schema"."tree_structure" where kpi_tree_id = '{}';""".format(lighthouse_context_,kpi_tree_id_), engine)
           nodes_list = nodes_list.values.tolist()   
#          raw_accounts_df.set_index('acc_id', inplace=True)
#       dispatcher.utter_message(text=f"{product_family_}")
           dispatcher.utter_message(text=f"Which node do you want me to check? You can type its full name. Here's the list of nodes identified within {kpi_tree_id_} KPI Tree ID : \n{nodes_list}.")
           return []
        else:
           nodes_list = pd.read_sql("""SELECT node_name FROM "incedo_{}_schema"."tree_structure" where kpi_tree_id in (SELECT kpi_tree_id from "incedo_{}_schema"."kpi_tree_master" where kpi_tree_name = '{}');""".format(lighthouse_context_,kpi_tree_name_), engine)            
           nodes_list = nodes_list.values.tolist()   
#          raw_accounts_df.set_index('acc_id', inplace=True)
#       dispatcher.utter_message(text=f"{product_family_}")
           dispatcher.utter_message(text=f"Which node do you want me to check? You can type its full name. Here's the list of nodes identified within {kpi_tree_name_} KPI Tree : \n{nodes_list}.")
           return []


class GetAllInsightsInThatNode(Action):
    def name(self) -> Text:
        return "action_talk_to_lighthouse_get_all_insights_in_that_node"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        lighthouse_context_ = tracker.get_slot("lighthouse_context")
        product_family_ = tracker.get_slot("product_family")
        kpi_tree_id_ = tracker.get_slot("kpi_tree_id")
        node_name_ = tracker.get_slot("node_name")
        kpi_tree_name_ = tracker.get_slot("kpi_tree_name")
        cohort_df = pd.read_sql("""SELECT * FROM "incedo_{}_schema"."cohort_analyser_daily" where kpi_tree_id = '{}';""".format(lighthouse_context_,kpi_tree_id_), engine)	
        msg_ = ''
        for ent_ in cohort_df.entity.unique():
            max_cohorts = cohort_df[(cohort_df['entity'] == ent_) & (cohort_df['kpi_value_current'] == cohort_df.loc[cohort_df['entity'] == ent_].kpi_value_current.max())]['segments']
            max_cohorts_value = cohort_df[(cohort_df['entity'] == ent_) & (cohort_df['kpi_value_current'] == cohort_df.loc[cohort_df['entity'] == ent_].kpi_value_current.max())]['kpi_value_current']*100
            msg_ = msg_ + "\n\nFor {} cohort type, the following cohorts have the maximum Roll Rate of {}% \n".format(ent_,max_cohorts_value.values[0]) + str(list(max_cohorts.values))
        dispatcher.utter_message(text=msg_)
        return []



        
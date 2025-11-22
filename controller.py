from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import utils.switch as switch
from datetime import datetime
from collections import deque
from io import StringIO

import pandas as pd
import os

from utils.machine_learning import MachineLearning

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):

        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.flow_stats_buffer = deque(maxlen=1)  # Keep only latest stats

        self.ml = MachineLearning()
        model_path = "trained_flow_model.joblib"
        dataset_path = 'dataset/FlowStatsfile.csv'
        dataset_name = os.path.basename(dataset_path)
        desired_model_type = 'KNN'
        
        if os.path.exists(model_path):
            self.logger.info("Checking if saved model matches desired model type and dataset...")
            self.ml.load_model(model_path)
            
            # Compare model type and dataset name (without loading current dataset)
            if (self.ml.dataset_name == dataset_name and
                self.ml.model_type == desired_model_type):
                self.logger.info(f"Model is compatible with {dataset_name} and model type ({desired_model_type}). Using saved model.")
            else:
                mismatch_reasons = []
                if self.ml.dataset_name != dataset_name:
                    mismatch_reasons.append(f"Dataset mismatch - Saved: {self.ml.dataset_name}, Current: {dataset_name}")
                if self.ml.model_type != desired_model_type:
                    mismatch_reasons.append(f"Model type mismatch - Saved: {self.ml.model_type}, Desired: {desired_model_type}")
                
                self.logger.info(f"Model is incompatible:")
                for reason in mismatch_reasons:
                    self.logger.info(f"  - {reason}")
                
                self.logger.info("Retraining model with current dataset and model type...")
                # Load dataset only when training is needed
                self.ml.load_dataset(dataset_path)
                start = datetime.now()
                self.ml.train_model(desired_model_type)
                end = datetime.now()
                self.logger.info(f"Training time: {(end-start)}")
                self.ml.save_model(model_path)
        else:
            self.logger.info("No saved model found. Training new model...")
            # Load dataset only when training is needed
            self.ml.load_dataset(dataset_path)
            start = datetime.now()
            self.ml.train_model(desired_model_type)
            end = datetime.now()
            self.logger.info(f"Training time: {(end-start)}")
            self.ml.save_model(model_path)

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(2)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now().timestamp()
        body = ev.msg.body
        
        flows_data = []

        for stat in sorted([flow for flow in body if (flow.priority == 1)], key=lambda flow:
            (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):
        
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            # Reset protocol-specific fields for each flow
            icmp_code = -1
            icmp_type = -1
            tp_src = 0
            tp_dst = 0
            
            if ip_proto == 1:  # ICMP
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
            elif ip_proto == 6:  # TCP
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']
            elif ip_proto == 17:  # UDP
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
          
            # Calculate per-second and per-nanosecond metrics safely
            try:
                packet_count_per_second = stat.packet_count / stat.duration_sec if stat.duration_sec > 0 else 0
                packet_count_per_nsecond = stat.packet_count / stat.duration_nsec if stat.duration_nsec > 0 else 0
            except (ZeroDivisionError, TypeError) as e:
                self.logger.warning(f"Error calculating packet rate: {e}")
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
                
            try:
                byte_count_per_second = stat.byte_count / stat.duration_sec if stat.duration_sec > 0 else 0
                byte_count_per_nsecond = stat.byte_count / stat.duration_nsec if stat.duration_nsec > 0 else 0
            except (ZeroDivisionError, TypeError) as e:
                self.logger.warning(f"Error calculating byte rate: {e}")
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
            
            flows_data.append({
                'timestamp': timestamp,
                'datapath_id': ev.msg.datapath.id,
                'flow_id': flow_id,
                'ip_src': ip_src,
                'tp_src': tp_src,
                'ip_dst': ip_dst,
                'tp_dst': tp_dst,
                'ip_proto': ip_proto,
                'icmp_code': icmp_code,
                'icmp_type': icmp_type,
                'flow_duration_sec': stat.duration_sec,
                'flow_duration_nsec': stat.duration_nsec,
                'idle_timeout': stat.idle_timeout,
                'hard_timeout': stat.hard_timeout,
                'flags': stat.flags,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'packet_count_per_second': packet_count_per_second,
                'packet_count_per_nsecond': packet_count_per_nsecond,
                'byte_count_per_second': byte_count_per_second,
                'byte_count_per_nsecond': byte_count_per_nsecond
            })
        
        # Store in buffer for next prediction cycle
        if flows_data:
            self.flow_stats_buffer.append(flows_data)

    def flow_predict(self):
        try:
            if not self.flow_stats_buffer:
                return
            
            # Get latest flows data
            flows_data = self.flow_stats_buffer[0]
            
            # Convert to DataFrame for prediction
            df = pd.DataFrame(flows_data)
            y_flow_pred = self.ml.predict(df)

            # Count predictions using NumPy for efficiency
            import numpy as np
            legitimate_traffic = np.sum(y_flow_pred == 0)
            ddos_traffic = np.sum(y_flow_pred != 0)

            self.logger.info(f"y_flow_pred : {y_flow_pred}")        

            self.logger.info("------------------------------------------------------------------------------")
            if len(y_flow_pred) > 0:
                legitimate_ratio = (legitimate_traffic / len(y_flow_pred)) * 100
                
                if legitimate_ratio > 80:
                    self.logger.info("legitimate traffic ...")
                else:
                    self.logger.info("ddos traffic detected ...")
                    # Find victim from DDoS flows (when prediction != 0)
                    ddos_indices = np.where(y_flow_pred != 0)[0]
                    if len(ddos_indices) > 0:
                        # Use first DDoS flow to identify victim
                        victim_idx = ddos_indices[0]
                        victim = int(df.iloc[victim_idx]['ip_dst'].split('.')[-1]) % 20
                        self.logger.info("victim is host: h{}".format(victim))
            
            self.logger.info("------------------------------------------------------------------------------")

        except IndexError as e:
            self.logger.error(f"Error accessing prediction data: {e}")
        except Exception as e:
            self.logger.error(f"Error in flow prediction: {e}")

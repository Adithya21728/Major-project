    import flwr as fl
    from typing import List, Dict, Tuple, Optional, Union
    from flwr.server.strategy import FedAvg
    from flwr.server.client_proxy import ClientProxy
    from flwr.common.typing import Scalar
    from flwr.common import EvaluateRes, FitRes  # Import from flwr.common instead
    from flwr.server.server import ServerConfig  # Add this import


    class CustomFedAvg(FedAvg):
        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation metrics using weighted average."""
            if not results:
                return None, {}
            
            # Print what's being received for debugging
            print(f"Round {server_round}: Received {len(results)} evaluation results")
            
            total_examples = sum(r.num_examples for _, r in results)
            aggregated_metrics = {}
            
            
            # Handle case where no metrics exist
            if not results or not results[0][1].metrics:
                return None, {}
                
            for key in results[0][1].metrics.keys():
                weighted_sum = sum(
                    r.metrics[key] * r.num_examples for _, r in results
                )
                aggregated_metrics[key] = weighted_sum / total_examples
                
            print(f"Aggregated metrics: {aggregated_metrics}")
            return None, aggregated_metrics

    if __name__ == "__main__":
        print("Starting FL server...")
        strategy = CustomFedAvg(
            fraction_fit=1.0,
            min_fit_clients=3,
            min_available_clients=3,
        )
        
        # Start Flower server
        fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # Use ServerConfig object instead of dict
        strategy=strategy,
    )
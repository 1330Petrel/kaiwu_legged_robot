from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from pprint import pprint

def main(args=None):
    if args.eth is not None:
        ChannelFactoryInitialize(0, args.eth)
    else:
        ChannelFactoryInitialize(0)
        
    robot_state = RobotStateClient()
    robot_state.Init()
    
    service_name = "sport_mode"
    # _, service_list = robot_state.ServiceList()
    # pprint(f"service list: {service_list}")

    code = robot_state.ServiceSwitch(service_name, False)
    if code == 0:
        print(f"service {service_name} closed")
    else:
        raise Exception(f"service {service_name} failed to close with return code: {code}")
    # import time
    # time.sleep(15.0)

    # _, service_list = robot_state.ServiceList()
    # pprint(f"service list: {service_list}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eth', type=str)
    args = parser.parse_args()
    
    main(args)
    


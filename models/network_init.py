import pandapower as pp
import pandapower.networks as nw

def get_cigre_lv_network():
    # 1. 加载 CIGRE 低压分配网
    net = nw.create_cigre_network_lv()
    
    # 2. 论文中提到有 10 个 active prosumers 和 5 个 passive consumers
    # 我们先看看默认网络有多少负荷
    print(f"默认网络负荷数量: {len(net.load)}")
    
    # 3. 运行基础潮流计算，确保网络是通的
    try:
        pp.runpp(net)
        print("网络初始化成功，基础潮流计算完成。")
    except:
        print("潮流计算失败，请检查网络连接。")
        
    return net

if __name__ == "__main__":
    network = get_cigre_lv_network()
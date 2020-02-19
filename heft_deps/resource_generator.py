from heft_deps.resource_manager import Resource, Node, SoftItem


class ResourceGenerator:

     @staticmethod
     def r(list_flops):
        result = []
        res = Resource("res_0")
        for flop, i in zip(list_flops, range(len(list_flops))):
            node = Node(res.name + "_node_" + str(i), i, res, [SoftItem.ANY_SOFT])
            node.flops = flop
            result.append(node)
        res.nodes = result
        return [res]
import hello_helpers.hello_misc as hm

class MyNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)

    def main(self):
        hm.HelloNode.main(self, 'my_node', 'my_node', wait_for_first_pointcloud=False)

        # my_node's main logic goes here
        self.move_to_pose({'joint_head_tilt': 0.0}, blocking=True)
        self.move_to_pose({'joint_head_pan': -1.5}, blocking=True)
        self.move_to_pose({'joint_lift': -1.8}, blocking=True)
        # self.move_to_pose({'gripper_aperture': 1.0}, blocking=True)

node = MyNode()
node.main()
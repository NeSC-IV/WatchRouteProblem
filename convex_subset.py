import matplotlib.pyplot as plt
import math
import copy
from functools import reduce
import operator
class Node(object):
    def __init__(self, num, x, y):
        self.v_num = num
        self.x = x
        self.y = y
        self.left = None
        self.right = None

class DoublyLinkedList():
    def __init__(self):
        self.head=None
        self.tail=None

    def append_k(self,node):
        if(self.head is None):
            self.tail = self.head = node
        else:
            node.right = self.tail
            self.tail.left = node
            self.tail = node

    def append_tail(self,node):
        if(self.head is None):
            self.tail = self.head = node
            return

        if(node.left is not None):
            node.right=self.head
            self.head.left = node
            return

        temp = self.head
        while(temp.right is not None):
            temp = temp.right
        node.left = temp
        temp.right=node

    
    def kplot(self):
        x = []
        y = []
        if(self.head == self.tail and self.head.right is not None and self.tail.left is not None):
            temp = self.head.left
            x.append(self.head.x)
            y.append(self.head.y)
            while temp != self.head:
                x.append(temp.x)
                y.append(temp.y)
                temp=temp.left
            x.append(x[0])
            y.append(y[0])
        else:
            temp=self.head
            while temp is not None:
                x.append(temp.x)
                y.append(temp.y)
                temp=temp.left
        return x,y
        
    def first_reflex_node(self):
        temp = self.head.right
        node = None
        while(temp!= self.head): #reflex test is always for a circular list
            if(ccw(temp.left,temp,temp.right) <0 ):# we will remove all straight edges. degenerate, ignore for now
                node = temp
                break
            temp = temp.right
        return node

    def reset_v_num(self,node):
        temp = self.head.right
        while(temp!=self.head):
            if temp == node:
                break
            temp = temp.right

        if(temp == node):
            i = 0
            node.v_num = i
            self.head = node
            temp = temp.right
            while(temp!=node):
                i+=1
                temp.v_num = i
                temp = temp.right
        else:
            print("the reflex edge vertex doesn't exist.")

def is_reflex(node):
    return ccw(node.left,node,node.right) <0 


#division by zero error, robust
def intersection(a,b,c,d):
    dtr = ((b.x-a.x)*(d.y-c.y)-(b.y-a.y)*(d.x-c.x))
    if(math.isclose(dtr,0)):
        print("lines are parallel.degenerate case")
        return None
    r = ((a.y-c.y)*(d.x-c.x)-(a.x-c.x)*(d.y-c.y))/dtr
    s = ((a.y-c.y)*(b.x-a.x)-(a.x-c.x)*(b.y-a.y))/dtr
#    if((r >0 or math.isclose(r,0)) and (r<1 or math.isclose(r,1)) and (s>0 or math.isclose(s,0)) and (s<1 or math.isclose(s,1))):
    if(r >=0 and r<=1  and s>=0 and s<=1 ):
        return (a.x+r*(b.x-a.x),a.y+r*(b.y-a.y))
    elif(is_between(a,c,b)):
        return (c.x,c.y)
    elif(is_between(a,d,b)):
        return (d.x,d.y)
    else:
        return None

def ccw(a,b,c):
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


def inf_coord(a,b, along_b = False):
    inf_d = 40
    v = (b.x - a.x, b.y - a.y)
    v_mod = math.sqrt(v[0]**2+v[1]**2)
    u = (v[0]/v_mod,v[1]/v_mod)
    if(along_b):
        p = (a.x + inf_d*u[0],a.y + inf_d*u[1])
    else:
        p = (a.x - inf_d*u[0],a.y - inf_d*u[1])
    return p

def distance(a,b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def is_between(a,c,b):
    return math.isclose(distance(a,c) + distance(c,b), distance(a,b))


def GetKernel(polygon):

    inpt = [(point[0],point[1]) for point in list(polygon.exterior.coords)]
    inpt.pop()

    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), inpt), [len(inpt)] * 2))
    inpt = (sorted(inpt, key=lambda inpt: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, inpt, center))[::-1]))) % 360, reverse=True))

    P = DoublyLinkedList()
    K = DoublyLinkedList()


    for i,pt in enumerate(inpt):
        node = Node(i,pt[0],pt[1])
        P.append_tail(node)

    #hack to avoid multiple ifs
    P.append_tail(node)


    r_node = P.first_reflex_node()
    if(r_node is None): #todo
        x_list = [i[0] for i in inpt]
        y_list = [i[1] for i in inpt]
        return x_list,y_list

    # P.reset_v_num(r_node) #r_node is the self.head node
    vi = r_node



    F1 = inf_coord(vi,vi.right) #1st edge. infinity-r_node
    L1 = inf_coord(vi,vi.left) #2nd edge. r_node-infinity


    F1 = Node(0,F1[0],F1[1])
    L1 = Node(1,L1[0],L1[1])

    K.append_k(F1)
    K.append_k(copy.deepcopy(vi))
    K.append_k(L1)


    vi = vi.right
    i = 1
    while(i<=len(inpt)-2):
        if(is_reflex(vi)): #vertex i is reflex
            vi_next_inf = inf_coord(vi,vi.right)
            vi_next_inf = Node(0,vi_next_inf[0],vi_next_inf[1])
            left_test = ccw(vi_next_inf,vi.right, F1)
            if(left_test<0 or math.isclose(left_test,0)): #F1 lies on or right of the vi+1 to inf line
                p = None
                q = None
                wt1 = F1
                wt2 = F1.left
                while(wt1!=L1):
                    p = intersection(wt2,wt1,vi.right,vi_next_inf)
                    if(p is not None):
                        w_d = Node(0,p[0],p[1])
                        break
                    wt1 = wt2
                    wt2 = wt2.left

                ws2 = F1
                ws1 = F1.right
                while(ws1 is not None):
                    q = intersection(ws2,ws1,vi.right,vi_next_inf)
                    if(q is not None):
                        w_dd = Node(0,q[0],q[1])
                        break
                    ws2 = ws1
                    ws1 = ws1.right
                if(q is None):
                    tail_end2 = K.tail
                    tail_end1 = tail_end2.right
                    head_end = K.head

                    if((ccw(tail_end1,tail_end2,vi_next_inf)  * ccw(vi.right,vi_next_inf, head_end)) > 0): #slope is comprised bw the slopes of the two half lines
                        wt2.right = w_d
                        w_d.left = wt2
                        w_d.right = vi_next_inf
                        vi_next_inf.left = w_d
                        #vi_next_inf.right = None
                        K.head = vi_next_inf
                    else:
                        wr2 = K.tail
                        wr1 = K.tail.right
                        while(True): #warning
                            q = intersection(wr1,wr2,vi.right,vi_next_inf)
                            if(q is not None):
                                w_dd = Node(0,q[0],q[1])
                                break
                        wt2.right = w_d
                        w_d.left = wt2
                        w_d.right = w_dd
                        w_dd.left = w_d
                        w_dd.right = wr1
                        wr1.left = w_dd
                        K.tail = K.head = wt2 # K is bounded now. should head be wt2?
                else:
                    w_d.left = wt2
                    wt2.right = w_d
                    w_d.right = w_dd
                    w_dd.left = w_d
                    w_dd.right = ws1
                    ws1.left = w_dd
                    if(K.head==K.tail):
                        K.head = K.tail = w_d

                if(q is None):
                    F1 = vi_next_inf
                else:
                    F1 = w_dd
                #case 11
                w1 = L1
                w2 = L1.left
                while(w2 is not None):
                    if(ccw(vi.right,w1,w2)>0):
                        L1 = w1
                        break
                    w1 = w2
                    w2 = w2.left

            else:
                #case 12 F1
                w1 = F1
                w2 = F1.left
                while(w2 is not None):
                    if(ccw(vi.right,w1,w2)<0):
                        F1 = w1
                        break
                    w1 = w2
                    w2 = w2.left
                #case 11 L1
                w1 = L1
                w2 = L1.left
                while(w2 is not None):
                    if(ccw(vi.right,w1,w2)>0):
                        L1 = w1
                        break
                    w1=w2
                    w2=w2.left
            
        else:
            vi_next_inf = inf_coord(vi,vi.right,along_b=True)
            vi_next_inf = Node(0,vi_next_inf[0],vi_next_inf[1])
            if(ccw(vi,vi_next_inf,L1)>0): # L1 is to the left. Ki+1 = Ki
                #case 12 F1
                w1 = F1
                w2 = F1.left
                while(w2 is not None):
                    if(ccw(vi.right,w1,w2)<0):
                        F1 = w1
                        break
                    w1 = w2
                    w2 = w2.left
                if(K.head == K.tail and K.head.right is not None and K.tail.left is not None): #bounded
                    #case 11 L1
                    w1 = L1
                    w2 = L1.left
                    while(w2 is not None):
                        if(ccw(vi.right,w1,w2)>0):
                            L1 = w1
                            break
                        w1=w2
                        w2=w2.left
            else:
                p = None
                q = None
                wt2 = L1
                wt1 = L1.right
                while(wt2!=F1):
                    p = intersection(wt2,wt1,vi,vi_next_inf)
                    if(p is not None):
                        w_d = Node(0,p[0],p[1])
                        break
                    wt2 = wt1
                    wt1 = wt1.right

                ws1 = L1
                ws2 = L1.left
                while(ws2 is not None):
                    q = intersection(ws1,ws2,vi,vi_next_inf)
                    if(q is not None):
                        w_dd = Node(0,q[0],q[1])
                        break
                    ws1 = ws2
                    ws2 = ws2.left
                if(q is None):
                    tail_end2 = K.tail
                    tail_end1 = tail_end2.right
                    head_end = K.head
                    if((ccw(tail_end1,tail_end2,vi_next_inf)  * ccw(vi,vi_next_inf, head_end)) > 0): #slope is comprised bw the slopes of the two half lines
                        wt1.left = w_d
                        w_d.right = wt1
                        w_d.left = vi_next_inf
                        vi_next_inf.right = w_d
                        #vi_next_inf.left = None
                        K.tail = vi_next_inf #vi_next_inf is the head?
                    else:
                        wr1 = K.head
                        wr2 = wr1.left
                        while(True):
                            q = intersection(wr1,wr2,vi,vi_next_inf)
                            if(q is not None):
                                w_dd = Node(0,q[0],q[1])
                                break
                            wr1 = wr2
                            wr2 = wr2.left
                        wr2.right = w_dd
                        w_dd.left = wr2
                        w_dd.right = w_d
                        w_d.left = w_dd
                        w_d.right = wt1
                        wt1.left = w_d
                        K.tail = K.head = wr2
                else:
                    w_d.left = w_dd
                    w_d.right = wt1
                    w_dd.right = w_d
                    w_dd.left = ws2
                    wt1.left = w_d
                    ws2.right = w_dd
                    if(K.head==K.tail):
                        K.head = K.tail = w_d

                #update F1 and L1
                if(q is None):
                    L1 = vi_next_inf
                    if(is_between(vi,vi.right,w_d)):
                        #case 12 F1
                        w1 = F1
                        w2 = F1.left
                        while(w2 is not None):
                            if(ccw(vi.right,w1,w2)<0):
                                F1 = w1
                                break
                            w1 = w2
                            w2 = w2.left
                    else:
                        F1= w_d
                else:
                    if(is_between(vi,vi.right,w_d)):
                        #case 12 F1
                        w1 = F1
                        w2 = F1.left
                        while(w2 is not None):
                            if(ccw(vi.right,w1,w2)<0):
                                F1 = w1
                                break
                            w1 = w2
                            w2 = w2.left
                    else:
                            
                            F1 = w_d
                    if(is_between(vi,vi.right,w_dd)):
                        L1 = w_dd
                    else:
                        w1 = w_dd
                        w2 = w_dd.left
                        while(w2 is not None):
                            if(ccw(vi.right,w1,w2)>0):
                                L1 = w1
                                break
                            w1=w2
                            w2=w2.left

        vi=vi.right
        i+=1

    x_list , y_list = K.kplot()
    x_list.pop()
    y_list.pop()
    return x_list,y_list


def GetReflexPointList(polygon):
    
    reflex_list = []
    point_list = list(polygon.exterior.coords)
    point_list.pop()
    n = point_num = len(point_list)
    while point_num < 2 * n:
        a = point_list[(point_num-1) % n]
        b = point_list[(point_num) % n]
        c = point_list[(point_num+1) % n]
        if ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) < 0:
            reflex_list.append(point_list[point_num % n])
        point_num += 1
    return reflex_list
if __name__ == '__main__':

    # inpt = [(3.588709677419355, 3.4199134199134198), (6.4919354838709671, 5.6926406926406923), (13.608870967741936, 5.4220779220779232), (15.866935483870968, 5.2326839826839837), (16.693548387096772, 10.50865800865801), (11.068548387096772, 11.212121212121213), (9.3951612903225801, 17.353896103896105), (3.9516129032258061, 18.354978354978357), (8.125, 13.511904761904763), (2.520161290322581, 10.535714285714286), (1.55241935483871, 7.1807359307359313), (1.028225806451613, 1.3365800865800868)]
    # polygon = random_polygons_generate.GetPolygon(5)
    # pointsOfPolygon = list(polygon.exterior.coords)
    inpt = [(point[0] * 20+0.1,point[1]*20+0.1) for point in pointsOfPolygon]
    print(inpt)

    x_list = [i[0] for i in inpt]
    x_list.append(inpt[0][0])
    y_list = [i[1] for i in inpt]
    y_list.append(inpt[0][1])
    plt.plot(x_list,y_list)

    inpt.pop()
    x_list,y_list = GetKernel(inpt)
    x_list.append(x_list[0])
    y_list.append(y_list[0])
    plt.plot(x_list,y_list)

    plt.axis([0,1,0,1])
    plt.show()
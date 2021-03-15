import numpy as np
from transformations import rotate, rotation_matrix, eucl

# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
# from graph import axisEqual3D 

class robot_model:
    def __init__(self, configfile):
        self.pos_canal = [-13.3, 6.2]  # position of the instrument channel wrt camera channel. z towards scene x down y right


        d = {}
        with open(configfile) as f:
            for line in f:
                (key,val) = line.split("=")
                d[key] = val

        self.diameters = np.array(d['diameters'][:-1].split(',')).astype(float)
        self.d = float(d['d'][:-1])
        self.lengths_l = np.array(d['L_left'][:-1].split(',')).astype(float)
        self.lengths_r = np.array(d['L_right'][:-1].split(',')).astype(float)
        self.dth = float(d['dth'][:-1])
        self.n_sec = int(d['n_sec'][:-1])
        self.active = np.array(d['active'][:-1].split(',')).astype(int)


        self.setOrigin([0,0,0],[0,0,1],[1,0,0]) #sam
        self.r = np.zeros(self.n_sec) #radi

        
    def setOrigin(self, P, nz, nx):
        self.P = P
        self.z = nz
        self.z /= np.linalg.norm(self.z)
        self.x = nx
        self.x /= np.linalg.norm(self.x)
        self.y = np.cross(nz,nx)

        
    def setCurvature(self, r):
        self.r = np.asarray(r)

    def computeShape(self, q):
        # sets parameters accoding to specified joint values 

        radii = [1000000,1000000,1000000,1000000,1000000] # intially all sections have very high radius of curvature (=straight)
        rotation = [0,0,0,0,0] # intial own rotation is 0 for all
        if self.arm == "L":
            lengths = list(self.lengths_l)
            self.lengths = self.lengths_l
        else:
            lengths = list(self.lengths_r)
            self.lengths = self.lengths_r
        for i in range(self.n_sec):
            if self.active[i]==1:
                if np.abs(q[2]) > 0:
                    bend_ang = q[2]
                    radii[i] = self.lengths[i]/bend_ang*self.diameters[i]
                rotation[i] = q[1]*np.pi/180

        self.setCurvature(radii)
        lengths[0] = q[0]

        # compute 3d points
        P = self.get_fullshape(lengths, rotation)
        return P


    def get_fullshape(self, d, th):
        # d = length
        # th = rotation
        corrected_flag = False
        P = []
        tan = []
        norm = []
        segm = 1
        (dest_P, dest_tan, dest_norm), _ = self.computeCircArc(self.P,self.z, self.x, th[0], self.r[0], d[0],str(segm))


        start_P0 = dest_P[-1]


        P.append(dest_P[:-1,:])

        tan.append(dest_tan)
        norm.append(dest_norm)
        for i in range(1,self.n_sec - 1):
            segm += 1

            (dest_P, dest_tan, dest_norm), corrected = self.computeCircArc(dest_P[-1],tan[i-1][-1], norm[i-1], th[i], self.r[i], d[i],str(segm))

            P.append(dest_P)
            tan.append(dest_tan)
            norm.append(dest_norm)
            if corrected: corrected_flag = True

        if corrected_flag:
            n_diff = 40 - len(P[1])
            if n_diff > 0:
                p_interp = np.linspace(start_P0,P[1][0],39+n_diff)
                P[0] = p_interp[:39]
                P[1] = np.concatenate([p_interp[39:],P[1]], axis=0)
            elif P[1][-1,2] < 0 and P[2][-3,2] > 0:
                ind_pos = np.where(np.diff(P[2][:, 2] > 0))[0][0] + 1
                n_neg = np.sum(P[2][:,2]<0)
                p_interp0 = np.linspace(start_P0,np.array([0,0,0]),39)
                p_interp1 = np.linspace(np.array([0,0,0]), P[2][ind_pos], 40)
                P[0] = p_interp0
                P[1] = p_interp1
                p_interp2 = np.linspace(P[2][ind_pos],P[2][ind_pos+1],n_neg)
                P[2] = np.concatenate([P[2][ind_pos:ind_pos+1], p_interp2, P[2][ind_pos+1:]], axis=0)
            else:
                print("err")
        return P
  

  
        
    def computeCircArc(self,start_P, start_tan, start_n, th, r, l, segm):
        R_th1 = rotation_matrix(th,start_tan)

        u = rotate(R_th1, start_n) 
        n = np.cross(u,start_tan)



        num = 40
        l_ = np.linspace(0,int(l),num=num)


        a = l_*1.0 / r
        center = start_P - r*u

        dest_P = center + r * ( np.cos(a).reshape(a.shape[0],1) * np.array(u).reshape(1,3) + np.sin(a).reshape(a.shape[0],1) * np.cross(n,u).reshape(1,3) )

 

        dest_tan = []
        for al in a:
            dest_tan.append(rotate(rotation_matrix(al,n),start_tan))
        dest_tan = np.array(dest_tan)
        dest_norm = np.cross(dest_tan[-1],n)



        correctn = True
        corrected = False
        if  start_P[2] < 0 and segm == "2" and correctn:
            corrected=True
            extracted_length = l - eucl(start_P, [0, 0, 0])
            not_extracted_length = eucl(start_P, [0, 0, 0])
            if self.arm == "L":
                p = np.polyfit([l - 12.23, l - 6., l - 10.729,l - 0.96,0.,15.],[6.12,8.43,7.26,l,3.5,l],3)
            if self.arm == "R":
                p = [1. / l, 0, 0]


            dist_from_end = np.polyval(p,not_extracted_length)
            if l - dist_from_end < extracted_length:
                dist_from_end = l - extracted_length
            if dist_from_end >= l:
                dist_from_end = l - extracted_length
            if not start_P[2] <= -l + 1./3:
                acc_length = 0
                dest_P_new = []
                dest_tan_new = []
                f = False
                for c in range(len(dest_P) - 1,0,-1):
                    acc_length += eucl(dest_P[c],dest_P[c-1])
                    if acc_length >= dist_from_end and acc_length <= dist_from_end + extracted_length:
                        dest_P_new.append(dest_P[c])
                        dest_tan_new.append(dest_tan[c])
                        f = True
                    elif f:
                        dest_P_new.append(dest_P[c])
                        dest_tan_new.append(dest_tan[c])
                        f = False
                if len(dest_P_new) == 0: # approximation of borderline case
                    dest_P_new.append(dest_P[c])
                    dest_tan_new.append(dest_tan[c])
                dest_P_new = dest_P_new[::-1]
                dest_tan_new = dest_tan_new[::-1]
                dest_P = []
                #print("extracted: "+str(extracted_length)+", distance from end: "+str(dist_from_end))
                for i in range(len(dest_P_new)):
                    dest_P.append(np.array(dest_P_new[i]) - np.array(dest_P_new[0]))
                dest_P = np.array(dest_P)
                dest_tan = np.array(dest_tan_new)
                dest_norm = np.cross(dest_tan[-1], n)
            elif start_P[2] <= -l + 1./3 and start_P[2] >= -l:
                acc_length = eucl(dest_P[1],dest_P[0])
                dest_P_new = [dest_P[0]]
                dest_tan_new = [dest_tan[0]]
                dist_from_start = l - dist_from_end
                c = 1
                while acc_length <= dist_from_start:
                    dest_P_new.append(dest_P[c])
                    c += 1
                    acc_length += eucl(dest_P[c], dest_P[c - 1])
                dest_P = []
                #print("extracted: "+str(extracted_length)+", distance from end: "+str(dist_from_end))
                for i in range(len(dest_P_new)):
                    dest_P.append(np.array(dest_P_new[i]) - np.array(dest_P_new[0]))
                dest_P = np.array(dest_P)
                dest_tan = np.array(dest_tan_new)
                dest_norm = np.cross(dest_tan[-1], n)
            else:
                #print("completely retracted")
                P_last = np.array([start_P[0], start_P[1], start_P[2] + l])
                x = np.linspace(start_P[0],P_last[0],num)
                y = np.linspace(start_P[1],P_last[1],num)
                z = np.linspace(start_P[2],P_last[2],num)
                dest_P_new = np.array([np.array(a) for a in zip(x,y,z)])
                dest_tan_new = np.array([np.array(a) for a in list([start_tan])*len(dest_P_new)])
                dest_P = np.array(dest_P_new)
                dest_tan = np.array(dest_tan_new)
                dest_norm = np.cross(dest_tan[-1], n)


        return(dest_P, dest_tan, dest_norm), corrected



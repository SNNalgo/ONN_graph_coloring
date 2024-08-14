import numpy as np
import matplotlib.pyplot as plt

def cordic_16_32(theta, atan_table, frac_bits): # 3.13 bit decimal representation with 32 bit integers for handling multiplications
    a = np.int32(0)
    x = np.int32(1<<frac_bits)
    y = np.int32(0)
    p2i = np.int32(1<<frac_bits)
    iters = atan_table.shape[0]
    for i in range(iters):
        if a<theta:
            a = a + atan_table[i]
            x_new = x - np.int32((p2i*y)>>frac_bits)
            y_new = np.int32((p2i*x)>>frac_bits) + y
        else:
            a = a - atan_table[i]
            x_new = x + np.int32((p2i*y)>>frac_bits)
            y_new = -np.int32((p2i*x)>>frac_bits) + y
        p2i = p2i>>1
        x = x_new
        y = y_new
    return x,y #x is cos(theta), y is sin(theta)

def cordic_16_32_vec(theta, atan_table, frac_bits): # 3.13 bit decimal representation with 32 bit integers for handling multiplications
    a = np.int32(np.zeros_like(theta))
    sigma = np.int32(np.zeros_like(theta))
    
    x = np.int32(np.ones_like(theta))<<frac_bits
    y = np.int32(np.zeros_like(theta))
    
    x_new = np.int32(np.ones_like(theta))<<frac_bits
    y_new = np.int32(np.zeros_like(theta))
    
    #p2i = np.int32(1<<frac_bits)
    iters = atan_table.shape[0]
    for i in range(iters):
        #sigma[a<theta] = np.int32(1)
        #sigma[a>=theta] = np.int32(-1)
        #a = a + sigma*atan_table[i]
        #x_new = x - sigma*((p2i*y)>>frac_bits)
        #y_new = sigma*((p2i*x)>>frac_bits) + y
        #p2i = p2i>>1
        
        sigma[a<theta] = np.int32(0)
        sigma[a>=theta] = np.int32(-1)
        a = a + ((atan_table[i]^sigma) - sigma)
        x_new = x - (((y>>i)^sigma) - sigma)
        y_new = (((x>>i)^sigma) - sigma) + y
        
        x = x_new
        y = y_new
        #print("x dtype: ", x.dtype)
        #print("y dtype: ", y.dtype)
        #print("a dtype: ", a.dtype)
        
        #print("x_new dtype: ", x_new.dtype)
        #print("y_new dtype: ", y_new.dtype)
        #print("sigma dtype: ", sigma.dtype)
    return x,y #x is cos(theta), y is sin(theta)

if __name__ == "__main__":
    K = 0.6072529350088812561694
    frac_bits = 13 # 13 bits for fractional part + 3 bits for whole number part -> 16 bit signed representation
    atan_table = np.array([np.arctan2(1, 2**i) for i in range(frac_bits)])

    factor = np.int32(1<<frac_bits)
    atan_table = np.int32(factor*atan_table)
    K_fxp = np.int32(factor*K)
    fxp_pi = np.int32(np.pi*(1<<frac_bits))
    fxp_half_pi = np.int32(np.pi*(1<<(frac_bits-1)))

    test_angs = np.pi*np.arange(-180, 185, 5)/180
    test_angs_int = np.int32(factor*test_angs)
    test_angs_int_cp = np.int32(factor*test_angs)
    #Input angle values must lie between -pi/2 - pi/2
    #Following correction will give cos and sin with opposite sign for values outside -pi/2 -> pi/2
    test_angs_int[test_angs_int>fxp_half_pi] = -fxp_pi + test_angs_int[test_angs_int>fxp_half_pi]
    test_angs_int[test_angs_int<-fxp_half_pi] = fxp_pi + test_angs_int[test_angs_int<-fxp_half_pi]
    
    print("arc tan table (int): ", atan_table)
    
    cordic_cos, cordic_sin = cordic_16_32_vec(test_angs_int, atan_table, frac_bits)
    #correcting the sign for values outside -pi/2 -> pi/2
    cordic_cos[test_angs_int_cp>fxp_half_pi] = -cordic_cos[test_angs_int_cp>fxp_half_pi]
    cordic_sin[test_angs_int_cp>fxp_half_pi] = -cordic_sin[test_angs_int_cp>fxp_half_pi]
    cordic_cos[test_angs_int_cp<-fxp_half_pi] = -cordic_cos[test_angs_int_cp<-fxp_half_pi]
    cordic_sin[test_angs_int_cp<-fxp_half_pi] = -cordic_sin[test_angs_int_cp<-fxp_half_pi]
    #normalizing the value
    cordic_cos = (K_fxp*cordic_cos)>>frac_bits
    cordic_sin = (K_fxp*cordic_sin)>>frac_bits
    
    print("cordic cos dtype: ", cordic_cos.dtype)
    print("cordic sin dtype: ", cordic_sin.dtype)
    cordic_cos_fl = np.float32(cordic_cos)/factor
    cordic_sin_fl = np.float32(cordic_sin)/factor
    
    i = 0
    for theta in test_angs_int:
        #cordic_cos, cordic_sin = cordic_16_32(theta, atan_table, frac_bits)
        #cordic_cos = (K_fxp*cordic_cos)>>frac_bits
        #cordic_sin = (K_fxp*cordic_sin)>>frac_bits
        #cordic_cos_fl = np.float32(cordic_cos)/factor
        #cordic_sin_fl = np.float32(cordic_sin)/factor
        np_cos = np.cos(test_angs[i])
        np_sin = np.sin(test_angs[i])
        
        #print("cordic cosine: ", cordic_cos, "cordic sine: ", cordic_sin)
        #print("angle: ", test_angs[i]*180/np.pi, " np cosine: ", np_cos, "cordic cosine float: ", cordic_cos_fl, "diff: ", abs(np_cos - cordic_cos_fl))
        #print("angle: ", test_angs[i]*180/np.pi, " np sine: ", np_sin, "cordic sine float: ", cordic_sin_fl, "diff: ", abs(np_sin - cordic_sin_fl))
        
        print("cordic cosine: ", cordic_cos[i], "cordic sine: ", cordic_sin[i])
        print("angle: ", test_angs[i]*180/np.pi, " np cosine: ", np_cos, "cordic cosine float: ", cordic_cos_fl[i], "diff: ", abs(np_cos - cordic_cos_fl[i]))
        print("angle: ", test_angs[i]*180/np.pi, " np sine: ", np_sin, "cordic sine float: ", cordic_sin_fl[i], "diff: ", abs(np_sin - cordic_sin_fl[i]))
        i = i+1
    
    plt.figure()
    plt.plot(test_angs*180/np.pi, np.cos(test_angs))
    plt.plot(test_angs*180/np.pi + 1, cordic_cos_fl) #seperating to make the two visible
    plt.title('cosine')
    plt.legend(['numpy cosine', 'cordic cosine'])
    plt.savefig('cordic_cosine.png', dpi=300)
    plt.show()
    
    plt.figure()
    plt.plot(test_angs*180/np.pi, np.sin(test_angs))
    plt.plot(test_angs*180/np.pi + 1, cordic_sin_fl) #seperating to make the two visible
    plt.title('sine')
    plt.legend(['numpy sine', 'cordic sine'])
    plt.savefig('cordic_sine.png', dpi=300)
    plt.show()
    
# complex colormap util for image plots
# Barnett 12/5/23, based on z2rgb_kawski.m from 2007
using Colors

"""
    C = z2color(z) converts a complex number z into a Colors object
    which can be plotted as a pixel of a color image (heatmap).
"""
function z2color(z::Number)
    m = abs(z)
    if m==0.0
        L = 0.0; x=1.0; y=0.0;
    elseif isnan(z) || isinf(z)
        L = 1.0; x=1.0; y=0.0;
    else
        dyn = 3                        # larger gives wider dynamic range
        L = (1+tanh(log(m)/dyn))/2     # lightness (black through cols to white)
        #hue = (180/pi)*mod(angle(z),2pi)  # [0,360) deg; red = 0 phase
        #HSL(hue, 1.0, L)  # has ugly pi/3 kinks (not C^1) vs hue :(
        # HSL also has a kink at L=0.5, bad. Revert to my old z2rgb code... 
        x = real(z)/m; y = imag(z)/m
    end
    w = 1.22*(0.5-abs(L-0.5));  # see z2rgb_kawski; note has kink like L
    RGB(L+w*(-y/sqrt(6)+x/sqrt(2)), L+w*(-y/sqrt(6)-x/sqrt(2)), L+w*y*sqrt(2/3))
end

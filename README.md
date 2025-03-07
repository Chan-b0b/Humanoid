# Humanoid



hip pitch 88
hip roll 88
hip yaw 88
knee 139
ankle pitch 50
ankle roll 50

waist yaw 88
waist roll 50
waist pitch 50

shoulder pitch 25
shoulder roll 25
shoulder yaw 25
elbow 25
wrist roll 25
wrist pitch 5
wrist yaw 5

sit

[-0.25918102  0.02497864 -0.02166918  0.63990021 -0.42534393 -0.0328571
 -0.34970999  0.02148438 -0.01978111  0.65782213 -0.39432734 -0.03999678
 -0.00131607  0.          0.          0.28932786  0.12443042 -0.01850766
  0.98173189  0.07388496  0.12906027  0.00826913  0.29678202 -0.13680971
 -0.00452566  0.98307443 -0.13356638  0.09456134  0.01106704]

 [-2.75103569e-01 -3.97968292e-03  8.75723362e-03  5.30731678e-01
 -2.75199205e-01  1.03864884e-02 -1.98341370e-01 -9.28878784e-03
 -5.51891327e-03  4.53785419e-01 -3.02678019e-01  9.75576229e-03
 -5.12599945e-04  0.00000000e+00  0.00000000e+00  2.89879084e-01
  2.21047401e-01 -2.05450058e-02  9.79071379e-01  7.90741444e-02
  6.62207603e-02 -2.32618451e-02  2.90430546e-01 -2.18134642e-01
  2.93173790e-02  9.80425835e-01 -1.33853912e-01  4.08260822e-02
  4.54358831e-02]

[-1.47345066 -0.00718403  0.00342724  1.02432442  0.07828387  0.00851981
 -1.46544218 -0.01359272  0.00772572  1.02773309  0.05094006  0.02762892
  0.00662565  0.          0.         -0.17712224  0.13370633 -0.04223645
  1.08972168  0.07841492  0.12131357  0.06781825 -0.27715445 -0.06634247
 -0.51665974  1.20389557 -0.13319468  0.13681483  0.16846114]


 def check_foot_contact(sim):
    left_foot_contact = False
    right_foot_contact = False

    for contact in sim.data.contact[: sim.data.ncon]:  # Loop through active contacts
        geom1_name = sim.model.geom_id2name(contact.geom1)
        geom2_name = sim.model.geom_id2name(contact.geom2)

        # Check if foot is touching the ground
        if "left_foot" in [geom1_name, geom2_name] and "ground" in [geom1_name, geom2_name]:
            left_foot_contact = True
        if "right_foot" in [geom1_name, geom2_name] and "ground" in [geom1_name, geom2_name]:
            right_foot_contact = True

    return left_foot_contact, right_foot_contact
    

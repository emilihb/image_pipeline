#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# @author Emili Hernandez
# @date   October 2015
# 
# CSIRO Autonomous Systems Laboratory
# Queensland Centre for Advanced Technologies
# PO Box 883, Kenmore, QLD 4069, Australia
#
# Copyright (c) 2015, CSIRO
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sensor_msgs.msg
import cv2
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import tarfile


def normalize(angle):
    """Returns angle (in rads) in (-pi, pi] range ."""
    return angle + (2 * np.pi) * np.floor((np.pi - angle) / (2 * np.pi))


def compose(a, b):
    """6DoF composition between 'a' and 'b'

    :param a: [x, y, z, roll, pitch, yaw] row/column vector
    :param b: [x, y, z, roll, pitch, yaw] or [x, y, z] row/column vector
    :type a: 6-element array
    :type b: 3 or 6-element array
    :return: composition
    :rtype: 6-element array
    """
    # ensure row vector for internal access
    if len(a.shape) is 2:
        a = a.T[0]
    if len(b.shape) is 2:
        b = b.T[0]

    sr = np.sin(a[3])
    cr = np.cos(a[3])
    sp = np.sin(a[4])
    cp = np.cos(a[4])
    sy = np.sin(a[5])
    cy = np.cos(a[5])

    r = np.array([
        a[0] + b[0]*cp*cy + b[1]*(sp*sr*cy - sy*cr) + b[2]*(sp*cr*cy + sr*sy),
        b[0]*sy*cp + a[1] + b[1]*(sp*sr*sy + cr*cy) + b[2]*(sp*sy*cr - sr*cy),
        -b[0]*sp + b[1]*sr*cp + a[2] + b[2]*cp*cr])

    if len(b) is 6:
        r = np.r_[r, normalize(a[3:] + b[3:])]

    return r


def axis(f=1.):
    return np.array([[f, 0., 0.], [0., f, 0.], [0., 0., f]])


def camera(f=0.5, w=1.0, h=0.25):
    return np.array([
        [0., 0., 0.], [f, w/2, h/2], [f, -w/2, h/2], [0., 0., 0.],
        [0., 0., 0.], [f, w/2, -h/2], [f, -w/2, -h/2], [0., 0., 0.],
        [f, w/2, h/2], [f, w/2, -h/2],
        [f, -w/2, h/2], [f, -w/2, -h/2]
        ])


def parse_report_mono(strm, pattern="left-"):
    yml = yaml.load(strm)
    intrinsics = sensor_msgs.msg.CameraInfo()
    intrinsics.width = yml['image_width']
    intrinsics.height = yml['image_height']
    intrinsics.K = yml['K']
    intrinsics.D = yml['D']
    intrinsics.R = yml['R']
    intrinsics.P = yml['P']

    extrinsics = [(i, v['rvec'], v['tvec'], v['reprojection_error']) for i, v in yml.iteritems() if i[0:len(pattern)] == pattern]
    extrinsics = sorted(extrinsics, key=lambda x: x[0])

    reprojection_error = yml['reprojection_error']
    return intrinsics, reprojection_error, extrinsics


def compose_camera(p, coord):
    out = []
    for c in coord:
        r = compose(p, c)
        out.append(r)
    return np.asarray(out)


def rodrigues_to_euler(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec))
    euler = [
        np.arctan2(R[2, 1], R[2, 2]),
        np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)),
        np.arctan2(R[1, 0], R[0, 0])]
    return euler


def plot_extrinsics(extrinsics):
    cam = camera(1.5, 1., 0.5)
    # z- looking forward
    cam = compose_camera(np.array([0., 0., 0., 0., -np.pi/2, 0.]), cam)
    axx = axis(3)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot axis
    ax.plot([0, 1], [0, 0], [0, 0], 'r')
    ax.plot([0, 0], [0, 1], [0, 0], 'g')
    ax.plot([0, 0], [0, 0], [0, 1], 'b')

    name = []
    for (n, r, t, _) in extrinsics:
        euler = rodrigues_to_euler(r)
        new_cam = compose_camera(np.asarray(t+euler), cam)
        ax.plot(new_cam[:, 0], new_cam[:, 1], new_cam[:, 2])
        name.append(n)

    name = substract_letters(name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # no axis equal in 3d -> work around found in stack overflow
    trans = [t for (_, _, t, _) in extrinsics]
    trans.append(axx[0].tolist())
    trans.append(axx[1].tolist())
    trans.append(axx[2].tolist())
    trans = np.asarray(trans)
    max_range = (np.max(trans, axis=0) - np.min(trans, axis=0))
    mean = np.mean(trans, axis=0)

    ax.set_xlim(mean[0] - max_range[0], mean[0] + max_range[0])
    ax.set_ylim(mean[1] - max_range[1], mean[1] + max_range[1])
    ax.set_zlim(mean[2] - max_range[2], mean[2] + max_range[2])

    return fig


def plot_error(name, error_img, error):
    fig = plt.figure()

    idx = range(len(error_img))
    plt.bar(idx, error_img)
    plt.plot(idx, [error] * len(error_img), 'r--')
    plt.title("Mean reprojection error")
    plt.xlabel("image")
    plt.ylabel("pix")
    plt.xticks(idx, name)
    return fig


def substract_letters(strm):
    import re
    dstrm = []
    for n in strm:
        noletters = re.sub("\D", "", n)
        dstrm.append(str(int(noletters)))
    return dstrm


def main(argv):
    filename = "/tmp/calibrationdata.tar.gz"
    if len(argv) is 2:
        filename = argv[1]

    try:
        if tarfile.is_tarfile(filename):
            archive = tarfile.open(filename, 'r')
            strm = archive.extractfile('report.yml').read()
        else:
            strm = file(filename, 'r')

        intrinsics, error, extrinsics = parse_report_mono(strm)

        name = [n for (n, _, _, _) in extrinsics]
        err_img = [e for (_, _, _, e) in extrinsics]
        img_idx = substract_letters(name)

        plot_extrinsics(extrinsics)
        plot_error(img_idx, err_img, error)

        plt.show()
    except Exception as e:
        print "Exception:", e
        print "Usage: %s filename.tar.gz (or .yml file)" % argv[0]

if __name__ == '__main__':
    main(sys.argv)

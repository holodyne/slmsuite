.. _tips:

Tips
====

Computer Configuration
----------------------

We have a few suggestions regarding how to configure Windows installations for
experimental SLM usage.

Set "Screen Sleep" to "Never"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although SLMs often exist as virtual displays, the action of "Screen Sleep" on Windows
machines can interfere with these displays. In some cases, the SLM becomes unresponsive or
stuck in periods surrounding a sleep event on the physical display(s).
We suggest turning `screen sleep off
<https://support.microsoft.com/en-us/windows/how-to-adjust-power-and-sleep-settings-in-windows-26f623b5-4fcc-4194-863d-b824e5ea7679>`_.
If energy conservation is desired, consider turning the physical display(s) off manually.

Disable "Windows Aero Peek"
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upon hovering over a program in the taskbar, Windows Aero Peek hides all other windows
to make finding this program easier. However, Aero Peek can also hide the window being used to
project data onto the SLM in the virtual display, causing most power to
return to the zeroth order, and in some cases burning holes through devices
(this occurred several times before we realized what was happening).
We suggest `disabling Aero Peek
<https://answers.microsoft.com/en-us/windows/forum/all/how-to-completely-disable-aero-peek-in-windows-11/ec65a8de-6401-4be2-a0b3-ba0d29c6cfe4>`_.

Disable "Windows Update"
~~~~~~~~~~~~~~~~~~~~~~~~

Windows Updates are important to ensure security and stability. However, on an
experimental setup, these (surprise) updates can wreak havoc. We suggest
`disabling Windows Update <https://answers.microsoft.com/en-us/windows/forum/all/how-do-i-permanently-disable-automatic-windows-10/82e1e076-8dff-475e-8c5e-a2061d1a4c5a>`_
and scheduling a periodic external reminder to install new updates manually.

Building an SLM Setup
---------------------

There are a variety of ways to configure an SLM in a beamline, each with advantages and
disadvantages. In addition, there are known "best practices" to consider when aligning.
We're in the process of writing a full guide; however, in the interim, the alignment
tutorial `here <https://aomicroscopy.org/designing-and-aligning-optical-systems-incorporating-liquid-crystal-spatial-light-modulators-slms>`_ should get you started!
// A light tap of the phone's vibration motor, where a native app gives one (a version swiped to,
// a long press, a prompt sent). Android browsers have the Vibration API. Safari on an iPhone has
// none, but since iOS 17.4 a click on an `<input type="checkbox" switch>` plays the system's
// switch haptic, so a hidden one is clicked (the method of the MIT `ios-haptics` package, from
// cobalt.tools). Called from a tap or gesture handler; if a later iOS drops it, nothing happens.

export function haptic() {
  try {
    if (navigator.vibrate?.(8)) return;
    const label = document.createElement("label");
    label.ariaHidden = "true";
    label.style.display = "none";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.setAttribute("switch", "");
    label.appendChild(input);
    document.head.appendChild(label);
    label.click();
    label.remove();
  } catch {
    /* no haptics here */
  }
}

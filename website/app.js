"use strict";
(function () {
  window.addEventListener("load", init);

  function init() {
    fetch("ascii.txt")
      .then(response => response.text())
      .then(text => displayAscii(text));
  }

  function displayAscii(text) {
    let asciiDisplayBox = qs(".ascii-display");
    asciiDisplayBox.innerHTML += text;
  }

  /* --------------------------- Helper Functions --------------------------- */

  /**
   * Returns the element that has the ID attribute with the specified value.
   * @param {string} idName - element ID
   * @returns {object} DOM object associated with id (null if none).
   */
  function id(idName) {
    return document.getElementById(idName);
  }

  /**
   * Returns the array of elements that match the given CSS selector.
   * @param {string} selector - CSS query selector
   * @returns {object[]} array of DOM objects matching the query (empty if none).
   */
  function qsa(selector) {
    return document.querySelectorAll(selector);
  }

  /**
   * Returns the first element that matches the given CSS selector.
   * @param {string} selector - CSS query selector
   * @returns {object} the first DOM object matching the query (empty if none).
   */
  function qs(selector) {
    return document.querySelector(selector);
  }

  /**
   * Returns a new HTMLElement of the given type, but does not
   * insert it anywhere in the DOM.
   * @param {string} tagName - name of the typ of element to create
   * @returns {object} the newly-created HTML Element
   */
  function gen(tagName) {
    return document.createElement(tagName);
  }

})();


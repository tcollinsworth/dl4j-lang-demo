package com.daisyworks.xor;

public class TestGates {
    enum Alg { NOT, AND, OR, XOR, JKFF }
    boolean predict(Alg alg, boolean A, boolean B, boolean clk, boolean set, boolean rst, boolean Qprev) {
        switch (alg) {
            case NOT:
                return !A;
            case AND:
                return A && B;
            case OR:
                return A || B;
            case XOR:
                return A ^ B;
            case JKFF:
                return JKFF(A, B, clk, set, rst, Qprev);
            default:
                //Can't happen
                throw new UnsupportedOperationException();
        }
    }

    /**
     * Stateless, caller must provide current/previous state, method returns next state.
     */
    private boolean JKFF(boolean A, boolean B, boolean clk, boolean set, boolean rst, boolean Qprev) {
        if (rst) {
            return false;
        }
        if (set) {
            return true;
        }
        if (!clk || (!A && !B)) {
            return Qprev;
        }
        //clk /
        if (A && B) {
            return !Qprev;
        }
        if (A) {
            return true;
        }
        return false;
    }
}

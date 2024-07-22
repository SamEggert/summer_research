import os
from pathlib import Path
from dawdreamer.faust import FaustContext
from dawdreamer.faust.box import *


def construct_faust_code(mb_saw_dir):
    # Use the provided mb_saw_dir
    file_names = [os.path.splitext(f)[0] for f in os.listdir(mb_saw_dir) if f.endswith('.wav')]
    file_names.sort()

    num_wave_tables = 16
    num_files = len(file_names)
    wave_names = [file_names[i] for i in range(0, num_files, num_files // num_wave_tables)]

    # Initialize the string to be inserted
    insert_string = "wavetables ="
    for wave in wave_names:
        tmp = f"\n    wavetable(soundfile(\"param:{wave}[url:{{'{wave}.wav'}}]\",1)),"
        insert_string += tmp
    insert_string = insert_string[:-1] + ";"

    # Dynamically construct the Faust code
    dsp_content = f"""
    import("stdfaust.lib");
    // `sf` is a soundfile primitive
    wavetable(sf, idx) = it.lagrangeN(N, f_idx, par(i, N + 1, table(i_idx - int(N / 2) + i)))
    with {{
        N = 3; // lagrange order
        SFC = outputs(sf); // 1 for the length, 1 for the SR, and the remainder are the channels
        S = 0, 0 : sf : ba.selector(0, SFC); // table size
        table(j) = int(ma.modulo(j, S)) : sf(0) : !, !, si.bus(SFC-2);
        f_idx = ma.frac(idx) + int(N / 2);
        i_idx = int(idx);
    }};
    {insert_string}
    NUM_TABLES = outputs(wavetables);
    // ---------------multiwavetable-----------------------------
    // All wavetables are placed evenly apart from each other.
    //
    // Where:
    // * `wt_pos`: wavetable position from [0-1].
    // * `ridx`: read index. floating point [0 - (S-1)]
    multiwavetable(wt_pos, ridx) = si.vecOp((wavetables_output, coeff), *) :> _
    with {{
        wavetables_output = ridx <: wavetables;
        coeff = par(i, NUM_TABLES, max(0, 1-abs(i-wt_pos*(NUM_TABLES-1))));
    }};
    S = 2048; // the length of the wavecycles, which we decided elsewhere
    wavetable_synth = multiwavetable(wtpos, ridx)*env1*gain
    with {{
        freq = hslider("freq [style:knob]", 200 , 50  , 1000, .01 );
        gain = hslider("gain [style:knob]", .5  , 0   , 1   , .01 );
        gate = button("gate");
        wtpos = hslider("WT Pos [style:knob]", 0   , 0    , 1   ,  .01);
        ridx = os.hsp_phasor(S, freq, ba.impulsify(gate), 0);
        env1 = en.adsr(.01, 0, 1, .1, gate);
    }};
    replace = !,_;
    process = ["freq":replace, "gain":replace, "gate":replace -> wavetable_synth];
    """
    return dsp_content


def convert_faust_to_jax(dsp_content):
    with FaustContext():
        DSP_DIR1 = str(Path(os.getcwd()))
        argv = ['-I', DSP_DIR1]
        box = boxFromDSP(dsp_content, argv)
        module_name = 'FaustVoice'
        jax_code = boxToSource(box, 'jax', module_name, ['-a', 'jax/minimal.py'])

    custom_globals = {}
    exec(jax_code, custom_globals)  # security risk!
    return custom_globals[module_name]
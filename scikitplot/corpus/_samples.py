# scikitplot/corpus/_samples.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Small built-in public-domain text samples for deterministic Corpus usage.

Samples in this module are intentionally tiny enough to ship with the package
and require no network access. They are convenience data, not authoritative
scholarly editions.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "HAMLET_TEXT",
]

HAMLET_TEXT: Final[str] = """\
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK

HAMLET. O that this too too solid flesh would melt,
Thaw, and resolve itself into a dew!
Or that the Everlasting had not fix'd
His canon 'gainst self-slaughter! O God! God!
How weary, stale, flat, and unprofitable
Seem to me all the uses of this world!
Fie on't! ah, fie! 'Tis an unweeded garden
That grows to seed. Things rank and gross in nature
Possess it merely. That it should come to this!

POLONIUS. Yet here, Laertes? Aboard, aboard, for shame!
The wind sits in the shoulder of your sail,
And you are stay'd for. There- my blessing with thee!
Give thy thoughts no tongue,
Nor any unproportion'd thought his act.
Be thou familiar, but by no means vulgar;
Those friends thou hast, and their adoption tried,
Grapple them unto thy soul with hoops of steel.
Give every man thy ear, but few thy voice;
Take each man's censure, but reserve thy judgment.
This above all- to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.

GHOST. I am thy father's spirit,
Doom'd for a certain term to walk the night,
And for the day confin'd to fast in fires,
Till the foul crimes done in my days of nature
Are burnt and purg'd away.
I could a tale unfold whose lightest word
Would harrow up thy soul, freeze thy young blood,
Make thy two eyes, like stars, start from their spheres.

HAMLET. O all you host of heaven! O earth! What else?
And shall I couple hell? O, fie! Hold, hold, my heart,
And you, my sinews, grow not instant old,
But bear me stiffly up. Remember thee!
Ay, thou poor ghost, while memory holds a seat
In this distracted globe. Remember thee!

HAMLET. To be, or not to be- that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles,
And by opposing end them. To die- to sleep-
No more; and by a sleep to say we end
The heartache, and the thousand natural shocks
That flesh is heir to. 'Tis a consummation
Devoutly to be wish'd. To die- to sleep.
To sleep- perchance to dream: ay, there's the rub!
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.

OPHELIA. O, what a noble mind is here o'erthrown!
The courtier's, soldier's, scholar's, eye, tongue, sword;
The expectancy and rose of the fair state,
The glass of fashion and the mould of form,
The observed of all observers, quite, quite down!

HAMLET. Speak the speech, I pray you, as I pronounced it to
you, trippingly on the tongue. But if you mouth it,
as many of your players do, I had as lief the
town crier spoke my lines. Nor do not saw the air
too much with your hand, thus, but use all gently;
for in the very torrent, tempest, and, as I may say,
the whirlwind of passion, you must acquire and beget
a temperance that may give it smoothness.

KING. O, my offence is rank, it smells to heaven;
It hath the primal eldest curse upon't,
A brother's murder. Pray can I not,
Though inclination be as sharp as will.
My stronger guilt defeats my strong intent,
And, like a man to double business bound,
I stand in pause where I shall first begin,
And both neglect.

HAMLET. Alas, poor Yorick! I knew him, Horatio. A fellow
of infinite jest, of most excellent fancy. He hath
borne me on his back a thousand times; and now, how
abhorred in my imagination it is!
Where be your gibes now? Your gambols? Your songs?
Your flashes of merriment that were won't to set the table on a roar?

HAMLET. There's a divinity that shapes our ends,
Rough-hew them how we will.

HORATIO. That is most certain.

HAMLET. Not a whit, we defy augury. There's a special
providence in the fall of a sparrow. If it be now,
'tis not to come; if it be not to come, it will be
now; if it be not now, yet it will come.
The readiness is all.
"""

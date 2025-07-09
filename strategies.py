from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import json

# 1) Define tu tabla con las 11 estrategias y sus ejemplos completos
tabla = {
    "Self-Compassion": {
        "descripcion": "Animar a actuar con amabilidad y ternura hacia uno mismo, incluso al cometer errores o sentirse decaído.",
        "ejemplos": [
            "Self-compassion involves responding in a non-judgmental, understanding, and caring way to ourselves even when things are tough.",
            "The next time you are being hard on yourself, notice the language you use. Ask yourself, ‘How might I change this language if I were speaking to a friend?’",
            "✍ “What’s one compassionate thing you can say to yourself right now?” ✍",
            "One key to self-compassion is learning to tolerate it when you're not where you want to be yet. Reflect on whether your expectations for yourself need adjustment.",
            "Here's a story about self-compassion from another person using this texting program:\n\nLast month after I got an email saying that I didn't get the job I interviewed for (I went through 3 rounds of interviews!). I was devastated. I thought to myself, ‘I failed… again.’ […]"
        ]
    },
    "Thought Challenging": {
        "descripcion": "Apoyar al usuario en identificar patrones de pensamiento poco útiles y encontrar nuevas formas de interpretar las situaciones.",
        "ejemplos": [
            "Sometimes we can improve our mood and wellbeing by finding new ways to think about our problems.",
            "Today, try noticing your negative thoughts. See if there are any thoughts that automatically pop into your head and bring you down.",
            "✍ “What negative automatic thought pops into your head sometimes and brings you down?” ✍",
            "✍ “Or text back 'E' to see some examples of common negative automatic thoughts.” ✍",
            "Here's a message written by someone else using this texting program:\n\n‘Try writing down how you feel. It helped me identify what was distressing me. I realized a lot of my bad feelings came from the unkind and untrue things I was telling myself.’"
        ]
    },
    "Positive Activities": {
        "descripcion": "Ayudar al usuario a planificar y participar en actividades gratificantes.",
        "ejemplos": [
            "When we feel down, we underestimate how enjoyable something will be. You can counter this by doing more of what you might enjoy, even when your mind tells you it isn't going to be very fun.",
            "Make a plan to do one pleasant activity later today. Take notice of how you feel immediately before and after you do that activity.",
            "✍ “Text back the activity you are planning.” ✍",
            "✍ “Can you think of one thing that you used to love doing but that you don't do too much anymore? When you used to do this activity a lot, what did you like about it?” ✍",
            "Here's a message about helpful activities written by another person using this texting program:\n\n‘Every person is different. Different things will boost their mood. You are the expert on what works best for you.’"
        ]
    },
    "Valued Living": {
        "descripcion": "Ayudar al usuario a identificar sus valores más importantes y dar pasos guiados por ellos.",
        "ejemplos": [
            "Acting in ways that reflect your values can help you feel more connected to the things that matter to you.",
            "Some common areas that people often have values about include: relationships, work or school, social life, family, spirituality, health, community life, leisure/fun.",
            "✍ “Text back with one value you hold.” ✍",
            "✍ “What action can you do today to live out your values? What value does it connect to?” ✍",
            "Here's a story from someone who connected to their values to resolve a conflict:\n\nEvery day I have to work from home now, which is hard because I have a young kid. This has positive and negative parts to it. When I am working and my son interrupts with non-school questions I find I have been getting more and more angry about this. […]"
        ]
    },
    "Relaxation": {
        "descripcion": "Enseñar al usuario técnicas para reducir el estrés y alcanzar bienestar mediante ejercicios como estiramientos, respiración o mindfulness.",
        "ejemplos": [
            "One technique for physical relaxation is called progressive muscle relaxation: [link]. In PMR, we tense and then release different muscle groups in our bodies. Start by curling your toes and feet tightly, holding for 30 seconds, then releasing the tension. Notice the contrast between how your muscles feel when they're tensed and relaxed. Try this with different muscle groups, one at a time, moving up or down your body for 5 minutes.",
            "It's not always easy to relax. That's okay, stress is normal and an important part of being alive. Relaxation is about knowing how and when to change your level of arousal.",
            "✍ “Where do you experience tension in your body when you're stressed?” ✍",
            "Here's something you can do right now to help manage stress: Take a two-minute break from whatever you are doing. Instead of picking up another task, just take the full two minutes to be present. Check in with yourself. Notice your breath. Notice how your body feels right now. Notice, without judging, the thoughts that cross your mind."
        ]
    },
    "Problem Solving": {
        "descripcion": "Ayudar al usuario a elaborar y llevar a cabo un plan estratégico para resolver problemas.",
        "ejemplos": [
            "Good morning! Today we'll focus on evaluating different solutions to a problem. By picking more efficient solutions you can free up more time to do things you care about.",
            "Not every problem has a perfect solution, but thinking through problems in specific ways can help you identify the best solution for you.",
            "Today, revisit your list of possible solutions to a problem. Read each one and consider its positive and negative consequences. How well does it help you achieve your goals?",
            "After reviewing all the solutions, see if one stands out as having the most benefits, and the fewest unintended consequences.",
            "Here's a message about problem solving from someone using this texting program:\n\n‘Brainstorm as many solutions as you can, even if they seem totally absurd or implausible. Sometimes freeing your brain up to think creatively can help you land on an outside-the-box solution that's a good fit.’"
        ]
    },
    "Connectedness": {
        "descripcion": "Facilitar la resolución de conflictos, la búsqueda de apoyo o el fortalecimiento de lazos sociales.",
        "ejemplos": [
            "Whether we are introverts or extroverts, we all need relationships to thrive. Building and improving our connections with other people can bring a sense of purpose into our lives.",
            "When you have more positive contact with others, you are less likely to get caught in negative thought patterns about yourself and your life.",
            "✍ “Is there someone who tends to brighten your day? Today, you might reach out to see if they'd like to have coffee or some other kind of get together.” ✍",
            "✍ “Think of a time you felt close to someone else. What helped you connect to that person in that moment?” ✍",
            "✍ “How does your mood affect how you feel about socializing?” ✍"
        ]
    },
    "Daily Routine": {
        "descripcion": "Ayudar al usuario a establecer y mantener rutinas diarias regulares (horas de comida, sueño, socialización, etc.).",
        "ejemplos": [
            "Good morning! Today we'll send some messages about daily routines. This is all about bringing consistency to things like meal times, bed times, socializing, and other daily activities.",
            "Stability in our routines can keep us grounded and free up our time and energy to focus on other things.",
            "✍ “What are some ways you can create a pleasant wake up routine?” ✍",
            "Here's a story from someone whose inconsistent sleep routine was leading to other problems: …",
            "Here's a message about daily routines from someone else using this texting program:\n\n‘I feel more motivated and prepared the next day if I organize and plan out my day the day before. I try to do it every evening before bed. Honestly, it is not enjoyable, but it is so worth it because staying organized keeps me feeling in control.’"
        ]
    },
    "Willingness": {
        "descripcion": "Apoyar al usuario a experimentar y aceptar emociones o pensamientos desafiantes.",
        "ejemplos": [
            "Willingness is about finding ways to be open to experiences, even if they are sometimes uncomfortable. Without willingness, it can be intimidating to do the things that truly matter to you.",
            "For example, we might value trying new things, but we experience anxiety the first few times we try something new (like public speaking, dating, or making friends). Being willing to experience temporary anxiety allows us to stick with a new activity so it can enrich our lives.",
            "Don’t forget to allow an unpleasant thought, emotion, or physical sensation to hang around without trying to change it or struggle against it.",
            "✍ “Can you think of a thought or experience you avoided in the past, but that you no longer avoid? What made you more willing to recognize and accept that thought or experience?” ✍",
            "Coming into contact with painful or unpleasant emotions can be uncomfortable, but see what it's like to practice willingness by holding these experiences gently. What do you notice over time as you make space for all kinds of emotions, even ones that feel unpleasant?"
        ]
    },
    "Help-Seeking": {
        "descripcion": "Proporcionar información sobre fuentes de ayuda formales e informales y guiar en cómo acceder a ellas.",
        "ejemplos": [
            "There are lots of types of help including medication, therapy, and the informal support we get from other people. Finding the type of help that best fits your needs can take a little thought and effort, but it's often worth it!",
            "Learn about the different types of help here: [LINK]",
            "Think of one issue you haven't talked about much, but that you might want to share with someone.",
            "Identify someone you would be willing to share that issue with. It could be a friend, family member, someone on the internet who has gone through something similar, or someone else.",
            "✍ “When do you find it most challenging to open up to others? When do you find it easiest?” ✍"
        ]
    },
    "Gratitude": {
        "descripcion": "Invitar al usuario a reflexionar sobre aspectos de su vida por los que se siente agradecido y expresarlo.",
        "ejemplos": [
            "Good morning! Let's focus on gratitude today. Practicing gratitude doesn't mean everything in our lives is perfect. It means we recognize aspects of our lives we appreciate, big or small.",
            "Today, write a note to someone telling them what you appreciate about them. You can share your gratitude any way you want, like a text, a letter, or an email.",
            "Don't forget to notice feelings of gratitude when they come.",
            "Here's a message about gratitude from someone else using this texting program:\n\n‘It's sometimes hard to do, but noticing that you can be grateful for the hard stuff in life can change how you cope with it. Looking at challenges and hardship as experience I can grow from helps me see that all aspects of being human are important.’",
            "✍ “What's something you do every day that you don't pay much attention to, but that you feel thankful for?” ✍"
        ]
    }
}


# 2) Construye el system prompt inyectando la tabla como JSON
system_prompt = (
    "Eres Aileon-Buddy, un agente empático que conoce estas 11 estrategias de psicoeducación.\n"
    "Cada vez que recibas la instrucción “ENVÍA_MENSAJE”, debes:\n"
    "  1. Elegir aleatoriamente una de las 11 estrategias.\n"
    "  2. Elegir aleatoriamente *uno* de sus mensajes de ejemplo.\n"
    "  3. Devolver únicamente el nombre de la estrategia y el mensaje seleccionado.\n\n"
    f"ESTRATEGIAS = {json.dumps(tabla, ensure_ascii=False)}\n"
)

# 3) Inicializa el cliente apuntando a tinyllama:1.1b-chat
client = ChatOllama(model_name="tinyllama:1.1b-chat")

# 4) Envía la solicitud
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="ENVÍA_MENSAJE")
]

response = client(messages)
print("Respuesta del modelo:")
print(response.content)
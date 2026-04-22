from my_agents.triage_agent import triage_agent, make_handoff
from my_agents.menu_agent import menu_agent
from my_agents.order_agent import order_agent
from my_agents.reservation_agent import reservation_agent
from my_agents.complaints_agent import complaints_agent

# triage -> specialists
triage_agent.handoffs = [
    make_handoff(menu_agent),
    make_handoff(order_agent),
    make_handoff(reservation_agent),
    make_handoff(complaints_agent),
]

# specialists -> specialists (+ optional triage fallback)
menu_agent.handoffs = [
    make_handoff(order_agent),
    make_handoff(reservation_agent),
    make_handoff(complaints_agent),
    make_handoff(triage_agent),
]

order_agent.handoffs = [
    make_handoff(menu_agent),
    make_handoff(reservation_agent),
    make_handoff(complaints_agent),
    make_handoff(triage_agent),
]

reservation_agent.handoffs = [
    make_handoff(menu_agent),
    make_handoff(order_agent),
    make_handoff(complaints_agent),
    make_handoff(triage_agent),
]

complaints_agent.handoffs = [
    make_handoff(menu_agent),
    make_handoff(order_agent),
    make_handoff(reservation_agent),
    make_handoff(triage_agent),
]
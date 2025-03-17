obs, _ = env.get_observations()
timestep = 0
# simulate environment
while simulation_app.is_running():
    start_time = time.time()
    # run everything in inference mode
    with torch.inference_mode():
        # agent stepping
        actions = policy(obs)
        # env stepping
        obs, _, _, _ = env.step(actions)
    if args_cli.video:
        timestep += 1
        # Exit the play loop after recording one video
        if timestep == args_cli.video_length:
            break

    # time delay for real-time evaluation
    sleep_time = dt - (time.time() - start_time)
    if args_cli.real_time and sleep_time > 0:
        time.sleep(sleep_time)

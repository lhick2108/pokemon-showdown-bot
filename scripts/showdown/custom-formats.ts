// Installed into <showdown>/config/custom-formats.ts by scripts/start_server.ps1 (idempotent).
// Note: This is the list of formats

export const Formats: import('../sim/dex-formats').FormatList = [
	{
		section: "Bot testing",
	},
	{
		// Gen 9 Random Battle rules, but the players bring their own (packed) teams. scripts/gen_teams.js
		// produces the teams with the real gen9randombattle generator, so levels/sets are randbats ones.
		// No 'Obtainable' (would reject generator sets), no Team Preview, no team: 'random'.
		name: "[Gen 9] Random Battle Fixed",
		mod: 'gen9',
		challengeShow: true,
		searchShow: false,
		rated: false,
		ruleset: ['Species Clause', 'HP Percentage Mod', 'Cancel Mod', 'Sleep Clause Mod', 'Illusion Level Mod'],
	},
];

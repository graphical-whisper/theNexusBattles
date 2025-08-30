// ai_action_catalog.ts
export type ActionGroup = "offense" | "defense" | "support";

export const SPECIAL_KIND: Record<string, ActionGroup> = {
  // Tank
  GOLPE_ESCUDO: "offense",
  MANO_PIEDRA: "defense",
  DEFENSA_FEROZ: "defense",
  // Warrior Arms
  EMBATE_SANGRIENTO: "offense",
  LANZA_DIOSES: "offense",
  GOLPE_TORMENTA: "offense",
  // Mage Fire
  MISILES_MAGMA: "offense",
  VULCANO: "offense",
  PARED_FUEGO: "defense", // devuelve daño recibido
  // Mage Ice
  LLUVIA_HIELO: "offense",
  CONO_HIELO: "defense",  // reduce ataque enemigo
  BOLA_HIELO: "defense",  // reduce daño enemigo
  // Rogue Poison
  FLOR_LOTO: "offense",
  AGONIA: "offense",
  PIQUETE: "offense",
  // Rogue Machete
  CORTADA: "offense",
  MACHETAZO: "offense",
  PLANAZO: "offense",
  // Shaman (curaciones/buffs)
  TOQUE_VIDA: "support",
  VINCULO_NATURAL: "support",
  CANTO_BOSQUE: "support",
  // Medic
  CURACION_DIRECTA: "support",
  NEUTRALIZACION_EFECTOS: "support",
  REANIMACION: "support",
};

export const MASTER_KIND: Record<string, ActionGroup> = {
  "MASTER.TANK_GOLPE_DEFENSA": "offense",
  "MASTER.ARMS_SEGUNDO_IMPULSO": "support",
  "MASTER.FIRE_LUZ_CEGADORA": "defense",
  "MASTER.ICE_FRIO_CONCENTRADO": "defense",
  "MASTER.VENENO_TOMA_LLEVA": "offense",
  "MASTER.MACHETE_INTIMIDACION_SANGRIENTA": "offense",
  "MASTER.SHAMAN_TE_CHANGUA": "support",
  "MASTER.MEDIC_REANIMADOR_3000": "support",
};

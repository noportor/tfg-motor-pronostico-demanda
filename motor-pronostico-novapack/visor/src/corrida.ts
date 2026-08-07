/** La corrida activa vive en la URL (/#/:corrida/:vista): compartible y sin estado global. */

import { useParams } from "react-router-dom";

export function useCorridaActiva(): string {
  const { corrida } = useParams<{ corrida: string }>();
  return corrida ?? "";
}

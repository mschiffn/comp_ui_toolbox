function y = dB( x, n )

y = n * log10( abs( x ) / max( abs( x(:) ) ) );

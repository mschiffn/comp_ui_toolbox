function print_header( str_title, chr_line )
%
% print graphical header with specified title
%
% author: Martin F. Schiffner
% date: 2020-02-25
% modified: 2020-02-25
%

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure string scalar for str_title
	if ~( isstring( str_title ) && isscalar( str_title ) )
        errorStruct.message = 'str_title must be a string scalar!';
        errorStruct.identifier = 'print_header:NoSingleString';
        error( errorStruct );
    end

	% ensure nonempty chr_line
	if nargin < 2 || isempty( chr_line )
        chr_line = '=';
	end

	%----------------------------------------------------------------------
	% 2.) print header
	%----------------------------------------------------------------------
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( '\n' );
	fprintf( ' %s\n', repmat( chr_line, [ 1, 80 ] ) );
	fprintf( ' %s (%s)\n', str_title, str_date_time );
	fprintf( ' %s\n', repmat( chr_line, [ 1, 80 ] ) );

end % sequences

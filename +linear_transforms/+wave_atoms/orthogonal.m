%
% superclass for all orthogonal discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-02-04
%
classdef orthogonal < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthogonal( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin < 1 || isempty( size )
                size = 1;
            end

            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'orthogonal:InvalidSize';
                error( errorStruct );
            end

            % ensure positive integers for size
            mustBePositive( size );
            mustBeInteger( size );

            %--------------------------------------------------------------
            % 2.) create discrete wave atom transforms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wave_atoms.type( ones( size ) );

        end % function objects = orthogonal( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( orthogonals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.orthogonal
            if ~isa( orthogonals, 'linear_transforms.wave_atoms.orthogonal' )
                errorStruct.message = 'orthogonals must be linear_transforms.wave_atoms.orthogonal!';
                errorStruct.identifier = 'string:NoOrthogonalWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( orthogonals ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( orthogonals )

                strs_out( index_object ) = sprintf( "orthogonal-%d", orthogonals( index_object ).N_layers );

            end % for index_object = 1:numel( orthogonals )

        end % function strs_out = string( orthogonals )

	end % methods

end % classdef orthogonal < linear_transforms.wave_atoms.type

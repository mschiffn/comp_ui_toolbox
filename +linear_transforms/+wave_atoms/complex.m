%
% superclass for all complex discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-30
% modified: 2020-02-04
%
classdef complex < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = complex( size )

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
            objects@linear_transforms.wave_atoms.type( repmat( 4, size ) );

        end % function objects = complex( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( complexes )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.complex
            if ~isa( complexes, 'linear_transforms.wave_atoms.complex' )
                errorStruct.message = 'complexes must be linear_transforms.wave_atoms.complex!';
                errorStruct.identifier = 'string:NoComplexWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( complexes ) );

            % iterate complex discrete wave atoms
            for index_object = 1:numel( complexes )

                strs_out( index_object ) = sprintf( "complex-%d", complexes( index_object ).N_layers );

            end % for index_object = 1:numel( complexes )

        end % function strs_out = string( complexes )

	end % methods

end % classdef complex < linear_transforms.wave_atoms.type

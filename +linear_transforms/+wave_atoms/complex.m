%
% superclass for all complex discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-30
% modified: 2020-04-15
%
classdef complex < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = complex( pat )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 0, 1 );

            % ensure nonempty pat
            if nargin < 1 || isempty( pat )
                pat = 'p';
            end

            % superclass ensures valid pat

            %--------------------------------------------------------------
            % 2.) create complex discrete wave atoms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wave_atoms.type( pat, 4 * ones( size( pat ) ) );

        end % function objects = complex( pat )

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

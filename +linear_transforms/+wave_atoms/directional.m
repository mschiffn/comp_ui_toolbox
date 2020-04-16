%
% superclass for all directional discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-30
% modified: 2020-04-15
%
classdef directional < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = directional( pat )

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
            % 2.) create directional discrete wave atoms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wave_atoms.type( pat, 2 * ones( size( pat ) ) );

        end % function objects = directional( pat )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( directionals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.directional
            if ~isa( directionals, 'linear_transforms.wave_atoms.directional' )
                errorStruct.message = 'directionals must be linear_transforms.wave_atoms.directional!';
                errorStruct.identifier = 'string:NoDirectionalWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( directionals ) );

            % iterate directional discrete wave atoms
            for index_object = 1:numel( directionals )

                strs_out( index_object ) = sprintf( "directional-%d", directionals( index_object ).N_layers );

            end % for index_object = 1:numel( directionals )

        end % function strs_out = string( directionals )

	end % methods

end % classdef directional < linear_transforms.wave_atoms.type

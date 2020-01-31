%
% superclass for all Vaidyanathan wavelet parameters
%
%   "The Vaidyanathan filter gives an exact reconstruction, but does not
%	 satisfy any moment condition.  The filter has been optimized for
%	 speech coding." (see [1])
%
% REFERENCES:
%	[1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef vaidyanathan < linear_transforms.wavelets.type

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = vaidyanathan( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create Vaidyanathan wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size );

        end % function objects = vaidyanathan( varargin )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( vaidyanathans )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.vaidyanathan
            if ~isa( vaidyanathans, 'linear_transforms.wavelets.vaidyanathan' )
                errorStruct.message = 'vaidyanathans must be linear_transforms.wavelets.vaidyanathan!';
                errorStruct.identifier = 'MakeONFilter:NoVaidyanathanWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( vaidyanathans ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( vaidyanathans )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Vaidyanathan' );

            end % for index_object = 1:numel( vaidyanathans )

            % avoid cell array for single vaidyanathans
            if isscalar( vaidyanathans )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( vaidyanathans )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( vaidyanathans )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.vaidyanathan
            if ~isa( vaidyanathans, 'linear_transforms.wavelets.vaidyanathan' )
                errorStruct.message = 'vaidyanathans must be linear_transforms.wavelets.vaidyanathan!';
                errorStruct.identifier = 'string:NoVaidyanathanWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "Vaidyanathan", size( vaidyanathans ) );

        end % function strs_out = string( vaidyanathans )

	end % methods

end % classdef vaidyanathan < linear_transforms.wavelets.type
